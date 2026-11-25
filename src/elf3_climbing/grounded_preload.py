"""Demonstrate v4's preload using only actuator torques from the official supported handoff."""
import json
import mujoco as mj
import numpy as np
from .platform_task import PlatformAscent,preload_pose


def demonstrate(env, index):
    from elf3_forearm import get_pose, StanceInverseDynamics
    env.reset(options={"handoff":index})
    m,d=env.model,env.data
    x0=get_pose(m,d)
    end=preload_pose(m,env.handoffs[index][1],env.prior,loaded=True)
    controller=StanceInverseDynamics(m,d)
    old_torque=d.ctrl.copy()
    dt=m.opt.timestep
    duration=.85
    hold=longest=0
    max_drop=0.
    bad=""
    trajectory=[]
    min_arms=np.full(2,np.inf)
    min_floor=np.full(2,np.inf)
    base_feet=d.xpos[env.feet].copy()
    max_slip=0.
    for tick in range(round((duration+.3)/dt)):
        t=tick*dt
        u=min(t/duration,1.)
        if tick%4==0:
            q=x0+(end-x0)*u*u*(3.-2*u)
            qd=(end-x0)*(6.*u*(1.-u)/duration)
            torque=controller.torque(d,q,qd,38.)
            d.ctrl[:]=old_torque+(torque-old_torque)*min(t/.1,1.)
        mj.mj_step(m,d)
        loads,hit,wrong=env.contacts()
        slip=float(np.linalg.norm(d.xpos[env.feet,:2]-base_feet[:,:2],axis=1).max())
        max_slip=max(max_slip,slip)
        if hit or wrong or not np.isfinite(d.qpos).all() or slip>.02:
            bad="gripper_contact" if hit else "unplanned_contact" if wrong else "slip_or_fall"
            break
        drop=env.start_height-d.xpos[env.pelvis,2]
        physical=min(loads[:2])>3. and min(loads[6:])>8.
        if physical:
            max_drop=max(max_drop,drop)
        at_depth=physical and abs(drop-env.prior.drop)<.015
        hold=hold+1 if at_depth else 0
        longest=max(longest,hold)
        if t>=duration:
            min_arms=np.minimum(min_arms,loads[:2])
            min_floor=np.minimum(min_floor,loads[6:])
        if tick%10==0:
            trajectory.append(np.r_[d.time,d.qpos,d.qvel,d.ctrl,loads,drop])
    passed=not bad and hold*dt>=.2
    report=dict(yaw_deg=env.yaw,passed=bool(passed),failure=bad or (None if passed else "end_hold"),
                target_drop_m=env.prior.drop,max_supported_drop_m=float(max_drop),end_hold_s=hold*dt,
                max_hold_s=longest*dt,max_foot_slip_m=max_slip,
                final_window_min_forearm_N=min_arms.tolist() if np.isfinite(min_arms).all() else None,
                final_window_min_ground_foot_N=min_floor.tolist() if np.isfinite(min_floor).all() else None)
    state=np.empty(mj.mj_stateSize(m,mj.mjtState.mjSTATE_INTEGRATION))
    mj.mj_getState(m,d,state,mj.mjtState.mjSTATE_INTEGRATION)
    return report,state,np.asarray(trajectory)


def export(args,approach,signature):
    env=PlatformAscent(approach,signature,args.handoffs,args.motion)
    args.output.mkdir(parents=True,exist_ok=False)
    reports=[]
    for i,path in enumerate(args.handoffs):
        report,state,trajectory=demonstrate(env,i)
        reports.append(report)
        if report["passed"]:
            np.savez_compressed(args.output/path.name,preloaded=state,samples=trajectory,
                         columns="time,qpos,qvel,ctrl,loads[8],pelvis_drop",model_signature=env.signature,
                         motion_sha256=env.prior.sha, platform_height_m=env.top,
                         yaw=env.yaw, report=json.dumps(report))
    (args.output/"evaluation.json").write_text(json.dumps(reports,indent=2,allow_nan=False))
    print(json.dumps(reports,indent=2))
