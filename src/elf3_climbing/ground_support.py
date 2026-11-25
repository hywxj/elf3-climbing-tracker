"""External platform contacts of all colliding links except the two feet."""

import torch


def _collision_owners(robot):
    """Resolve the actual rigid body owning each enabled collision shape."""
    from pxr import Usd, UsdPhysics

    owners = set()
    for prim in Usd.PrimRange(robot, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            continue
        owner = prim
        while owner != robot and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        if owner == robot and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Unattached ELF3 collision shape {prim.GetPath()}")
        owners.add(str(owner.GetPath().MakeRelativePath(robot.GetPath())))
    return owners


class GroundSupport:
    def __init__(self, env):
        from isaaclab.sensors import ContactSensor, ContactSensorCfg

        ref = env.reference_np
        top = int(
            (ref["terrain_box_centers_w"][:, 2] + ref["terrain_box_half_sizes"][:, 2]).argmax()
        )
        robot = env.sim.stage.GetPrimAtPath("/World/envs/env_0/Robot")
        owners = _collision_owners(robot)
        feet = set(ref["site_bodies"][:2].tolist())
        if len(feet) != 2 or not feet.issubset(owners) or not owners - feet:
            raise ValueError("Cannot identify both soles and the other ELF3 colliders")
        self.views = []
        for body in sorted(owners - feet):
            if body in env.support_sensors:
                # These sensors already filter every terrain box in reference order.
                self.views.append((env.support_sensors[body], top))
                continue
            sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{body}",
                    filter_prim_paths_expr=[f"/World/envs/env_.*/Platform_{top}"],
                    # A history requests eager physics-step reads in Isaac Lab. These
                    # additional views only need the control-step sample near standing.
                    history_length=0,
                )
            )
            env.scene.sensors["ground_nonfoot_" + body] = sensor
            self.views.append((sensor, 0))

    def force(self, candidates):
        """Sum per-link platform force magnitudes, valid on candidate rows only."""
        force = torch.zeros_like(candidates, dtype=torch.float32)
        if candidates.any():
            for sensor, column in self.views:
                # Each pair is a robot link against only the upper platform.
                # Opposing forces on different links must not cancel each other.
                on_platform = sensor.data.force_matrix_w[:, 0, column]
                force += on_platform.norm(dim=1)
        return force * candidates
