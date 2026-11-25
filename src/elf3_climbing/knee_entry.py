"""Reference-terrain stage-one training for the bilateral forearm to right knee transition."""
import json
from pathlib import Path

import mujoco as mj
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .platform_task import PlatformAscent
from .protocol import BackwardCurriculum


class KneeEntry(PlatformAscent):
    """Phase-zero starts at a measured docking handoff. Later starts are explicit CSI resets."""
    def __init__(self, approach, signature, paths, motion):
        self.csi = BackwardCurriculum(phase=.9, window=24, threshold=.75, step=.1)
        self.aligned = None
        super().__init__(approach,signature,paths,motion)
        self.aligned = self.reference_states(motion)
        self.end = self.prior.frame[3]+6
        self.static_action = self.model.qpos0[self.qa].copy()
        self.static_action[self.wrist] = self.handoffs_wrist()
        self.action_range = np.maximum(self.static_action-self.limits[:,0],self.limits[:,1]-self.static_action)
        self.reset(seed=0)

    def handoffs_wrist(self):
        return float(self.home[self.wrist])

    def reference_states(self,motion):
        with np.load(motion,allow_pickle=False) as a:
            top, ground = a["terrain_box_centers_w"][:,2]+a["terrain_box_half_sizes"][:,2]
            half = a["terrain_box_half_sizes"][0]
            center = a["terrain_box_centers_w"][0]
            axes = Rotation.from_quat(a["terrain_box_quat_wxyz"][0],scalar_first=True).as_matrix()
            start = (a["base_pos_w"][0]-center) @ axes
            axis = np.argmax(abs(start[:2])/half[:2])
            into = -np.sign(start[axis])*axes[:,axis]
            basis = np.column_stack((into,np.cross([0.,0,1],into),[0,0,1]))
            xyz = (a["base_pos_w"]-center)@basis
            xyz[:,:2] += [self.front+half[axis],0.]
            # Foot-supported frames align to ground. Airborne/knee frames align to the top.
            lift = np.flatnonzero(a["takeoff_stage"]=="ASSISTED_KNEE_TUCK")
            weight = np.interp(np.arange(len(xyz)),[lift[0]-1,lift[-1]],[0.,1.])
            xyz[:,2] = a["base_pos_w"][:,2]-ground + weight*(self.top-(top-ground))
            poses = np.tile(self.model.qpos0,(len(xyz),1))
            poses[:,:3] = xyz
            poses[:,3:7] = (Rotation.from_matrix(basis.T)*Rotation.from_quat(
                a["base_quat_wxyz"],scalar_first=True)).as_quat(scalar_first=True)
            poses[:,self.qa[self.prior.motor]]=a["joint_pos"]
        vel = np.zeros((len(poses), self.model.nv))
        for i in range(1,len(poses)-1):
            mj.mj_differentiatePos(self.model,vel[i],self.prior.time[i+1]-self.prior.time[i-1],
                                   poses[i-1],poses[i+1])
        return poses,vel

    def reset(self,seed=None,options=None):
        super().reset(seed=seed,options=options)
        if self.aligned is None:
            return self.observe(),{}
        opts=options or {}
        self.start_phase=float(opts.get("phase",self.csi.phase))
        self.begin=int(round(self.prior.start+self.start_phase*(self.end-self.prior.start)))
        self.frame_time=self.prior.time[self.begin]
        m,d=self.model,self.data
        if self.start_phase>0.:
            poses,vel=self.aligned
            # Rotate the reset around the edge so yaw cases really do differ in the simulator.
            angle=np.deg2rad(self.yaw)
            rot=Rotation.from_euler("z",angle)
            origin=np.array([self.front,0.,0.])
            mj.mj_resetData(m,d)
            d.qpos[:]=poses[self.begin]
            d.qpos[:3]=origin+rot.apply(d.qpos[:3]-origin)
            d.qpos[3:7]=(rot*Rotation.from_quat(d.qpos[3:7],scalar_first=True)).as_quat(scalar_first=True)
            d.qvel[:]=vel[self.begin]
            d.qvel[:3]=rot.apply(d.qvel[:3])
            mj.mj_forward(m,d)
        self.target[:,1]=self.prior.xy[:,1]
        self.command=d.qpos[self.qa].copy()
        self.action[:]=np.clip((self.command[self.active]-self.static_action[self.active])/
                               self.action_range[self.active],-1,1)
        self.arm_anchors=self.arm_points()
        self.foot0=d.xpos[self.feet,2].copy()
        self.start_height=d.xpos[self.pelvis,2]
        self.stage=2 if self.begin>=self.prior.frame[2] else 1 if self.begin>=self.prior.frame[1] else 0
        self.actual_release=False
        self.was_grounded=self.start_phase==0.
        self.release_assumed=self.stage==2
        self.last_ground[:]=0
        self.deepest=0.
        self.peak=0.
        self.knee_streak=self.best_streak=self.both_air=0
        self.post_reset_steps=0
        return self.observe(),{"start_phase":self.start_phase}

    def target_action(self,q):
        return np.clip((q[self.active]-self.static_action[self.active])/self.action_range[self.active],-1.,1.)

    def step(self,action):
        m,d=self.model,self.data
        previous=self.action.copy()
        self.action[:]=np.clip(action,-1.,1.)
        desired=self.static_action.copy()
        desired[self.active]+=self.action_range[self.active]*self.action
        self.command+=.35*(np.clip(desired,*self.limits.T)-self.command)
        bad=grip=False
        for _ in range(self.substeps):
            d.ctrl[:]=np.clip(self.kp*(self.command-d.qpos[self.qa])-self.kd*d.qvel[self.va],
                              self.ctrl_min,self.ctrl_max)
            mj.mj_step(m,d)
            # Include brief wrist and gripper hits across all 2-ms physics steps.
            pairs=d.contact.geom[d.contact.dist<=0]
            grip |= bool(np.any(pairs==self.grip))
            surfaces=np.any((pairs==self.table)|(pairs==self.floor),axis=1)
            bad |= bool(np.any(self.wrist_geom[pairs[surfaces]]))
        loads,hit,unexpected=self.contacts()
        grip |= hit
        bad |= unexpected
        self.tick+=1
        feet=loads[6:]>8.
        arms=bool(np.all(loads[:2]>3.))
        self.was_grounded |= bool(np.all(feet))
        self.last_ground[feet]=self.tick
        if self.was_grounded and np.all(feet) and arms and not (grip or bad):
            self.deepest=max(self.deepest,self.start_height-d.xpos[self.pelvis,2])
        if self.stage==0 and self.deepest>=.65*self.prior.drop:
            self.stage=1
        if self.stage==1 and arms and np.all(feet) and d.qvel[2]>.03 and (
            self.deepest-(self.start_height-d.xpos[self.pelvis,2])>.008):
            self.push_tick=self.tick
        unloaded=not np.any(feet) and arms and np.all(d.xpos[self.feet,2]-self.foot0>.005)
        self.both_air=self.both_air+1 if unloaded and np.ptp(self.last_ground)<=4 and self.tick-self.push_tick<=8 else 0
        if self.stage==1 and self.both_air>=2:
            self.stage=2
            self.actual_release=True
        # A load on the table's riser, distal shin, left knee or a foot is never a right-knee goal.
        valid=arms and loads[2]>8. and loads[3]<3. and np.all(loads[4:]<8.) and not(grip or bad)
        if self.stage==2 and valid:
            self.knee_streak+=1
            self.peak=max(self.peak,loads[2])
        else:
            self.knee_streak=0
        self.best_streak=max(self.best_streak,self.knee_streak)
        self.frame_time+=self.dt
        ref=min(self.end,int(np.searchsorted(self.prior.time,self.frame_time)))
        poses,vel=self.aligned
        qerr=np.mean((d.qpos[self.qa]-poses[ref,self.qa])**2)
        knee_dist=np.linalg.norm(d.geom_xpos[self.knee[0]]-self.target[2])
        pose_reward=np.exp(-4*qerr)
        rot=Rotation.from_quat(d.qpos[3:7],scalar_first=True)
        ref_rot=Rotation.from_quat(poses[ref,3:7],scalar_first=True)
        orient_reward=np.exp(-2*(rot.inv()*ref_rot).magnitude()**2)
        reward=self.dt*(pose_reward+orient_reward+np.exp(-4*knee_dist)+float(arms)+
                       4.*float(valid)-.10*np.mean((previous-self.action)**2))
        success=self.knee_streak>=10
        reason="right_knee_supported" if success else ""
        if grip or bad or d.qpos[2]<.42 or not np.all(np.isfinite(d.qpos)):
            reason="gripper_contact" if grip else "unplanned_contact" if bad else "fall"
            success=False
        elif loads[3]>8.:
            reason="left_knee_contact"
            success=False
        timeout=self.tick>=110 and not reason
        if timeout:reason="timeout"
        reward+=3.*float(success)-2.*float(bool(reason) and not timeout and not success)
        info={"success":bool(success),"reason":reason,"start_phase":self.start_phase,"start_frame":self.begin,
              "measured_takeoff":self.actual_release,"assumed_takeoff_at_reset":self.release_assumed,
              "right_knee_hold_s":self.best_streak*self.dt,"right_knee_load_N":float(self.peak),
              "supported_preload_m":float(self.deepest),"elapsed_s":self.tick*self.dt}
        return self.observe(),float(reward),bool(reason and not timeout),bool(timeout),info

def suite(policy,env,phases):
    rows=[]
    for phase in phases:
        for i,(yaw,_) in enumerate(env.handoffs):
            ob,_=env.reset(options={"phase":phase,"handoff":i})
            for _ in range(110):
                action,_=policy.predict(ob,deterministic=True)
                ob,_,end,timeout,info=env.step(action)
                if end or timeout:
                    rows.append(dict(info,yaw_deg=yaw))
                    break
    return rows


def initial_imitation(policy,env):
    x,y=[],[]
    for i in range(len(env.handoffs)):
        for frame in range(env.prior.start,env.end+1):
            phase=(frame-env.prior.start)/(env.end-env.prior.start)
            ob,_=env.reset(options={"phase":phase,"handoff":i})
            x.append(ob)
            y.append(env.target_action(env.aligned[0][min(env.end,frame+2),env.qa]))
    x = torch.as_tensor(np.asarray(x, dtype=np.float32), device=policy.device)
    y = torch.as_tensor(np.asarray(y, dtype=np.float32), device=policy.device)
    params=list(policy.policy.mlp_extractor.policy_net.parameters())+list(policy.policy.action_net.parameters())
    opt=torch.optim.Adam(params,lr=.002)
    for _ in range(180):
        pred=policy.policy.get_distribution(x).distribution.mean
        loss=(pred-y).square().mean()
        opt.zero_grad();loss.backward();opt.step()


def experiment(args,approach,signature):
    torch.set_num_threads(1)
    env=KneeEntry(approach,signature,args.handoffs,args.motion)
    out=Path(args.output)
    contract={"model_signature":env.signature,"reference_sha256":env.prior.sha,"height_m":float(env.top),
              "task":"v4_right_knee_csi_v1","right_knee_continuous_hold_s":.2}
    checkpoint=args.checkpoint if args.cmd=="evaluate-knee" else args.resume
    if checkpoint and json.loads((checkpoint.parent/"contract.json").read_text())!=contract:
        raise ValueError("Incompatible knee-entry model, terrain or motion")
    out.mkdir(parents=True,exist_ok=False)
    if checkpoint:
        policy=PPO.load(checkpoint,env=env,device=args.device)
    else:
        policy=PPO("MlpPolicy",env,n_steps=512,batch_size=64,n_epochs=6,device=args.device,
                   learning_rate=.00015,seed=args.seed,target_kl=.02,
                   policy_kwargs={"net_arch":{"pi":[128,128],"vf":[128,128]},"log_std_init":-3.4})
    print(f"PPO device: {policy.device} | MuJoCo physics: CPU", flush=True)
    if args.cmd=="evaluate-knee":
        rows=suite(policy,env,[0.,.5,.65,.75,.8,.9])
        (out/"evaluation.json").write_text(json.dumps(rows,indent=2))
        print("evaluation",[(r["start_phase"],r["yaw_deg"],r["right_knee_hold_s"],r["success"]) for r in rows])
        return
    (out/"contract.json").write_text(json.dumps(contract,indent=2))
    if not checkpoint:
        initial_imitation(policy,env)
    validation=KneeEntry(approach,signature,args.handoffs,args.motion)
    phases=[0.,.5,.65,.75,.8,.9]
    def score(rows):
        return tuple(sum(r["success"] for r in rows if r["start_phase"]==p) for p in phases)+(
             sum(r["right_knee_hold_s"] for r in rows),)
    before=suite(policy,validation,phases)
    best=score(before)
    best_step=policy.num_timesteps
    history=[]
    policy.save(out/"best")
    class Monitor(BaseCallback):
        def _on_step(self):
            for done,info in zip(self.locals["dones"],self.locals["infos"]):
                if done:
                    env.csi.observe([info["success"]],[info["start_phase"]])
            if self.n_calls%2048==0:
                self.assess()
            return True

        def assess(self):
            nonlocal best,best_step
            phase=env.csi.phase
            rows=suite(self.model,validation,sorted(set(phases+[phase])))
            current=[r for r in rows if abs(r["start_phase"]-phase)<1e-8]
            rate=np.mean([r["success"] for r in current])
            proposed=score([r for r in rows if r["start_phase"] in phases])
            if proposed>best:
                best=proposed
                best_step=self.model.num_timesteps
                self.model.save(out/"best")
            env.csi.promote(phase,rate)
            history.append(dict(steps=self.model.num_timesteps,phase=phase,next_phase=env.csi.phase,
                                success_rate=float(rate)))
            (out/"progress.json").write_text(json.dumps(history,indent=2))
    monitor=Monitor()
    policy.learn(args.steps,callback=monitor,reset_num_timesteps=not bool(checkpoint))
    monitor.assess()
    policy.save(out/"policy")
    final=suite(policy,validation,phases)
    (out/"training.json").write_text(json.dumps(dict(contract=contract,seed=args.seed,device=str(policy.device),timesteps=policy.num_timesteps,
        best_step=best_step,handoffs=[str(p) for p in args.handoffs],curriculum=env.csi.state(),before=before,
        after=final),indent=2))
    print("knee entry complete",policy.num_timesteps,best_step,env.csi.phase,flush=True)
