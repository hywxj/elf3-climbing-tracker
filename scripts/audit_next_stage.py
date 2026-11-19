"""Read-only phase and contact geometry summary for the next training stage."""

import json
import numpy as np
from elf3_climbing.paths import DEFAULT_PACK
from elf3_climbing.reference import read_pack


def main():
    pack, manifest = read_pack(DEFAULT_PACK)
    phases = []
    for name in dict.fromkeys(pack["phase"]):
        ix = np.flatnonzero(pack["phase"] == name)
        first, last = int(ix[0]), int(ix[-1])
        phases.append(
            {
                "name": name,
                "frames": [first, last],
                "seconds": len(ix) / 50,
                "root_start": pack["root_pos"][first].tolist(),
                "root_end": pack["root_pos"][last].tolist(),
                "active_contacts_end": pack["site_names"][pack["contact_mask"][last]].tolist(),
                "feet_end": pack["site_pos"][last, :2].tolist(),
                "forearms_elbows_end": pack["site_pos"][last, 2:6].tolist(),
            }
        )
        if name == "FOREARM_HOLD":
            break
    print(
        json.dumps(
            {
                "reference": str(DEFAULT_PACK),
                "source_sha256": manifest["source_sha256"],
                "phases": phases,
                "terrain_centers": pack["terrain_box_centers_w"].tolist(),
                "terrain_half_sizes": pack["terrain_box_half_sizes"].tolist(),
                "scope": "Reference kinematics and geometric labels only, no physical acceptance",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
