"""Build the source-informed full ELF3 elbow-climbing GEOMETRIC DEMO.
No training, hardware commands, contact forces or dynamic simulation.
Seed poses are embedded from model-constrained IK. Frames are solved at 15 Hz,
interpolated to 30 Hz, collision/contact polished, and independently inspected.
Existing motion/model files are read-only. Outputs must use new paths.
"""

from project_paths import GRIPPER_CONFIG, MJCF, OUTPUT, TOOLS_DIR, URDF
import argparse, tempfile
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument(
    "--reuse-states",
    type=Path,
    help="Revalidate/export existing planner states instead of solving again",
)
args = parser.parse_args()
if args.output.exists() or args.output.with_suffix(".checks.json").exists():
    parser.error("Output already exists")
args.output.parent.mkdir(parents=True, exist_ok=True)
work = tempfile.TemporaryDirectory(prefix="elf3_full_demo_")
work_dir = Path(work.name)
import sys, json
from pathlib import Path
import numpy as np, mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(TOOLS_DIR))
from plan_knee_entry import create_context, KEYPOSES

p = create_context()
m = p.m
d = p.d
names = p.names
waist = m.body("waist_z_link").id
lo = np.r_[[-1.8, -0.5, 0.6], [-0.8, -0.3, -0.8], m.jnt_range[p.ji, 0]]
hi = np.r_[[0.5, 0.5, 2.1], [0.8, 1.8, 0.8], m.jnt_range[p.ji, 1]]
iy = 6 + names.index("l_wrist_y_joint")
lo[iy] = -1.309
hi[iy] = -1.3089
armgoal = [p.armpts @ p.R.T + p.world(p.edge + 0.37, y, p.top + 0.0329) for y in [0.23, -0.23]]


def source(i):
    xyz = p.R.T @ (p.a["base_pos_w"][i] - p.box["center"])
    xyz[2] = p.a["base_pos_w"][i, 2]
    rot = Rotation.from_matrix(
        p.R.T @ Rotation.from_quat(p.a["base_quat_wxyz"][i], scalar_first=True).as_matrix()
    ).as_rotvec()
    return np.r_[xyz, rot, p.a["joint_pos"][i]]


def features(x):
    p.setq(x)
    en, ss, vertices = p.geometry()
    return dict(
        pelvis=d.xpos[waist].copy(),
        knees=d.xpos[p.knees].copy(),
        feet=np.array([d.xpos[b] + d.xmat[b].reshape(3, 3) @ p.sole for b in p.feet]),
        feetrot=d.xmat[p.feet].reshape(2, 3, 3).copy(),
        rootrot=d.xmat[m.body("torso_link").id].reshape(3, 3).copy(),
        vertices=vertices,
        environment=en,
        selfdist=ss,
    )


kg = [m.geom("hiking_" + s + "_hip_z_link_0").id for s in ["l", "r"]]
sg = [m.geom("hiking_" + s + "_knee_y_link_0").id for s in ["l", "r"]]
fg = [m.geom("hiking_" + s + "_ankle_x_link_0").id for s in ["l", "r"]]


def localworld(t):
    return p.world(p.edge + t[0], t[1], p.top + t[2])


def kneegap(f, k):
    vs = np.vstack([f["vertices"][g] for g in [kg[k], sg[k]]])
    near = np.linalg.norm(vs - f["knees"][k], axis=1) < 0.105
    return float(vs[near, 2].min() - p.top) if near.any() else 1.0


def build_sequence():
    rows = [
        {
            "name": "STAND_GROUND",
            "x": [
                -1.4707522191058546,
                -9.563213189481249e-06,
                1.1012096002923244,
                3.917766279290145e-05,
                -3.2150774495781285e-05,
                -7.908516332927258e-07,
                -0.00011391416080159937,
                0.00011137072889709802,
                -2.5517511375764897e-06,
                -0.18713899645648735,
                0.037769888993840525,
                -0.007114765754429625,
                0.47095795288879455,
                -0.2847403985920317,
                -0.038601655823581306,
                -0.18742507059524996,
                -0.03813721628844895,
                0.007201838441218923,
                0.4715318042261747,
                -0.2850665423939286,
                0.03868292375322691,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.308900000004883,
                0.04402943535695793,
                -0.24997177368778528,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
        {
            "name": "WALK_LEFT_CLEAR",
            "x": [
                -1.3686480005032253,
                -0.06889822280354395,
                1.0598705793421201,
                0.0003045332103554962,
                0.0001300570936632712,
                5.972738462279731e-05,
                -0.4301351052508856,
                0.04696479897328082,
                -0.6274877638320957,
                -0.11409058960247119,
                0.2421527379981994,
                0.5331833757516888,
                1.1604422104535226,
                -0.4351890688653962,
                -0.1938086289608728,
                0.08432507528154361,
                -0.09134156535748453,
                0.6179081701002332,
                1.0342450668333643,
                -0.6411596865975817,
                -0.02287417308447002,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.3089000000000002,
                0.04402943535695793,
                -0.24997177368778536,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
        {
            "name": "WALK_LEFT_LAND",
            "x": [
                -1.2693330404298657,
                0.0004024094878051449,
                1.0412660284472357,
                -0.0006178248321120464,
                0.0006962134232662243,
                -0.0002544342746030058,
                -0.26931402727144765,
                -0.11998196194228315,
                -0.6219393984491248,
                -0.04733058046796128,
                0.31307146963154986,
                0.564437085819609,
                0.9505097662970521,
                -0.38703169250761077,
                -0.11214625079117034,
                0.15164485312405315,
                -0.03508743533914603,
                0.6315755806200227,
                0.8863147074275836,
                -0.6804321353356226,
                0.056792770612882126,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.3089000000000002,
                0.04402943535695793,
                -0.24997177368778542,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
        {
            "name": "WALK_RIGHT_CLEAR",
            "x": [
                -1.1801284083850363,
                0.06985119073648213,
                1.0622626994459445,
                -0.0023006121532953977,
                -0.00026982451496575867,
                -0.0006978653286939567,
                -0.3558555504171414,
                -0.26165299902813516,
                -0.013619152156855883,
                -0.07839659178618268,
                0.258082139243917,
                -0.017742916634398552,
                0.9894128043244996,
                -0.5781354265142595,
                0.01782100968925351,
                -0.21540188633583945,
                0.15972568397391265,
                -0.000834894626392303,
                1.2194781851566412,
                -0.5808599284174777,
                0.10501999026307778,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.3089000000000002,
                0.04402943535695793,
                -0.24997177368778525,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
        {
            "name": "APPROACH_READY",
            "x": [
                -1.080430932745287,
                -0.00037729217341061523,
                1.059417673584632,
                0.0012965689171045914,
                -4.00505598743309e-05,
                0.00033984135425024455,
                -0.23268789899681777,
                -0.06617269568713517,
                -0.01425224777533844,
                -0.003182517586190843,
                0.1233944255220729,
                0.0035418268865373584,
                0.8392430371279961,
                -0.6253032326714855,
                -0.053650989815130404,
                -0.005737270783152027,
                0.04085237030627066,
                0.017646181434736914,
                0.7994347128889595,
                -0.48853364812086236,
                0.02321646070958501,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.3089000000000002,
                0.04402943535695793,
                -0.24997177368778542,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
        {
            "name": "ARM_HOVER",
            "x": [
                -0.8134879948226017,
                0.0008429248020097094,
                1.0066671394722435,
                -0.0015992172409765091,
                0.9111556992714589,
                0.004079365179308532,
                -0.4898022290699942,
                -0.06252208884256313,
                0.12748281507952314,
                -0.47001241180509307,
                0.053354674661134506,
                -0.14869498901434192,
                0.3095999423297096,
                -0.2805977457796218,
                -0.04925964255696028,
                -0.33255231368055543,
                -0.0054545985005707434,
                -0.1451833008931388,
                0.17590890819642482,
                -0.2578387019542683,
                0.006628597314816492,
                -1.693177356766273,
                0.35605702638859715,
                -0.33264953568301064,
                0.7103485906968201,
                -0.4493359103118931,
                -1.3089999999999997,
                0.04339169564824723,
                -1.688504633572874,
                -0.3748362091064251,
                0.33924184913275707,
                0.7081354140451898,
                0.48490419485778913,
                -0.002305774907243322,
                -0.018647430410794398,
            ],
        },
        {
            "name": "FOREARMS",
            "x": [
                -0.7886248070048787,
                0.04393410209179146,
                0.9391838854354472,
                -0.14835403943698405,
                1.1069649751746238,
                0.14266725382731946,
                -0.48073261213863044,
                0.06292250020969058,
                -0.035477942202473266,
                -0.8302360695849131,
                0.04831697267781368,
                -0.15226740216810855,
                0.5654248289045131,
                -0.40761202400965246,
                -0.033206850975020664,
                -0.7375277279050615,
                -0.013747733620138442,
                -0.1218438638947471,
                0.49368041442406146,
                -0.35547992162209147,
                0.01690534379102882,
                -1.8438030373451006,
                0.005412912725929354,
                -0.22081275414987078,
                0.7384682528703812,
                0.08188977023818501,
                -1.3089671065361372,
                0.2128873202144382,
                -1.4838985038348722,
                -0.571578117739155,
                0.04823625888301721,
                0.4829114792980386,
                -0.0869054717865926,
                1.013345962186001,
                -0.6345230079206309,
            ],
        },
        {
            "name": "RAISE_AND_TUCK",
            "x": [
                -0.6168122606894079,
                0.011017915924450913,
                1.0265241102417755,
                -0.004970168411506717,
                1.5914023497902237,
                0.029521458386704183,
                -0.2922230984273236,
                0.16883390668359224,
                -1.0421046049243854,
                -1.9794691976798946,
                0.9121537195388575,
                -0.4868104067127597,
                0.9925644288830996,
                -0.7105409324636094,
                -0.03443136405249779,
                -2.2560122357875723,
                0.4414120039208786,
                -0.8153074230614337,
                1.2014282573569286,
                -0.2633959672936885,
                0.027442252887139838,
                -1.4276617054253264,
                0.25491149405549113,
                0.019156435454918866,
                -0.12572385274230485,
                0.019931942163534548,
                -1.3089999300976394,
                0.31431634777114154,
                -1.372511389566492,
                -0.3622286546780898,
                -0.08948909837738331,
                -0.16562050363301173,
                -0.15294983151425073,
                1.0139059313231318,
                -0.7772650765589163,
            ],
        },
        {
            "name": "RIGHT_KNEE",
            "x": [
                -0.5135939698523384,
                0.025535799478703745,
                0.9990534036657539,
                -0.004671689457404519,
                1.7999887341023166,
                0.13666401886807092,
                -0.399313073033067,
                -0.2030336803288031,
                -0.14482484918469662,
                -1.7783810019708306,
                0.09978161390586619,
                0.24513480559146492,
                0.8757121230684791,
                -0.7141520196814667,
                -0.04739159807186808,
                -2.0767791921454273,
                -0.11479545455268841,
                -0.16241702050167767,
                2.317195925896754,
                -0.2758859513745799,
                0.040883373193583795,
                -1.2655188799234363,
                0.15955500286988775,
                -0.004154171589591264,
                -0.5152760625193544,
                0.03389836451855162,
                -1.3089999643342334,
                0.2924950195188661,
                -1.0166233762310386,
                -0.46143221740713136,
                -0.4605150795478843,
                -0.5975529122103199,
                -0.020789964349931443,
                1.0806971774127292,
                -0.7822347504242516,
            ],
        },
        {
            "name": "LEFT_KNEE_CLEAR",
            "x": [
                -0.47315804371557535,
                0.004773557086900925,
                0.9791958049849806,
                -0.021520116567243293,
                1.799999998855923,
                0.09423197330658308,
                0.31595313065365666,
                -0.2616799292511443,
                0.2618943846259426,
                -2.608876375149207,
                0.0018501219891845797,
                0.5280059313331414,
                1.9417940884095437,
                -0.708428648653474,
                -0.0654526957396308,
                -2.788497607458761,
                -0.32317914490737054,
                -0.1416279477672288,
                2.269090159574477,
                -0.279053823214779,
                0.022268318663477855,
                -1.0759504299099683,
                0.252428344340907,
                0.17131864898451088,
                -0.6595260817421059,
                -0.015756958872503352,
                -1.3089999864091681,
                0.30889843986524457,
                -0.9232409451086139,
                -0.30282262048856595,
                -0.47508986387634594,
                -0.7229840945262398,
                -0.0537829137537617,
                1.0571885444562363,
                -0.7839368089679289,
            ],
        },
        {
            "name": "BOTH_KNEES",
            "x": [
                -0.44120472282957374,
                -0.0009783632130945525,
                0.941244932244441,
                -0.0006386616141220413,
                1.7998179386445587,
                -0.00997275494439436,
                -0.023871372936003686,
                -0.026387989910918633,
                -0.08645369613741517,
                -2.5981610041370446,
                0.0828882340604312,
                0.5281430328658643,
                2.3565345139989,
                -0.8675322897256054,
                0.2841922822645509,
                -2.4743698905956344,
                0.003463594169724595,
                -0.7209145979487483,
                2.4459382292692764,
                0.7830526823929265,
                0.24168440540042047,
                -0.7883743412649536,
                0.3165953505724086,
                0.41514208372786143,
                -0.8814520008172418,
                -0.0907296489165434,
                -1.3089999970627484,
                0.36117584641778544,
                -0.8052602739534344,
                -0.2935959410802673,
                -0.36626591524586655,
                -0.8938363376100893,
                -0.07454117725994712,
                1.0495158554304158,
                -0.7851803509175971,
            ],
        },
        {
            "name": "KNEEL_UP",
            "x": [
                -0.44026425013587733,
                0.0002994957098483966,
                1.3709411765218047,
                -0.0017636473619537087,
                0.46622586075724326,
                0.0007893108683006057,
                0.5219989992604995,
                -0.09006339494278603,
                -0.04818462541481365,
                -1.4636487846676742,
                0.24752170735638693,
                0.15981932714876432,
                2.1354356988968486,
                -0.5001921879664886,
                0.3426952331858188,
                -1.255015137516544,
                -0.021259807923259457,
                -0.0736279162074736,
                2.0782581252371606,
                0.784936618729404,
                0.3490699996922957,
                -0.25001416317567204,
                0.14999967440930487,
                0.35334930557010874,
                0.5000281276061151,
                -0.08986341971585574,
                -1.3089000000000002,
                0.04538263183398276,
                -0.24997588563296128,
                -0.15000569137161932,
                -0.10246909176782866,
                0.5000000908326369,
                0.010704010736618553,
                0.15063510768308042,
                0.02782426449586423,
            ],
        },
        {
            "name": "RIGHT_FOOT_CLEAR",
            "x": [
                -0.5106506201216091,
                0.019471217709075905,
                1.4079025648561292,
                -0.10375353727953666,
                0.3963539278647698,
                0.05058548241081942,
                0.097664184414676,
                -0.26128800415985154,
                0.20892413285480446,
                -1.31216273817805,
                0.2961969438032475,
                -0.7010047599717436,
                2.1357131243513123,
                -0.6117341335587344,
                0.3429194690439055,
                -2.2465664779699224,
                -0.9954693502731645,
                -0.5393230194416548,
                2.3873875046061754,
                -0.3993941947687004,
                -0.3472939242384732,
                -0.2500154970193212,
                0.15000673180503896,
                0.15916143099553598,
                0.5000249677482969,
                -0.09013227125590105,
                -1.3089000000000002,
                0.04425144054363193,
                -0.24997695418756866,
                -0.14999908852741223,
                0.038430044479211306,
                0.5000032572335777,
                0.01570456142892057,
                0.14631015883214302,
                0.04438418040809452,
            ],
        },
        {
            "name": "RIGHT_FOOT",
            "x": [
                -0.480809880128706,
                -0.0002595077013169489,
                1.413585876346967,
                -0.0017365729093575606,
                0.4541403234420878,
                0.002259570115812226,
                0.10093059626200283,
                -0.2308145402362371,
                0.15122566448941757,
                -1.141596444639249,
                0.23566142769469434,
                -0.29487448184781284,
                2.120879732573604,
                -0.7260502933632178,
                -0.3360096017770981,
                -1.95751246949013,
                -0.3866553940862471,
                -0.2526891499382918,
                2.0204678444017463,
                -0.5485189381127834,
                -0.1759532395084468,
                -0.25001388482962283,
                0.15000908440417982,
                0.0021062878279069,
                0.5000237648993976,
                -0.08965766464101356,
                -1.3089000000000002,
                0.045309567116732695,
                -0.2975584200217713,
                -0.09650293433816558,
                0.4907804343279302,
                0.46047072511129095,
                -0.018084839274269814,
                0.1361905662872375,
                0.04410724832694557,
            ],
        },
        {
            "name": "LEFT_FOOT_CLEAR",
            "x": [
                -0.3819393809623507,
                -0.09999057062097047,
                1.580919091686386,
                -2.981455697330827e-05,
                0.45319652399109195,
                6.399592037661853e-05,
                0.4827944895839909,
                -0.0023145918204331615,
                0.12124414902245474,
                -1.9639223949691222,
                0.06321477883682176,
                -0.26984952465887185,
                1.7033406125058423,
                -0.700227241095648,
                -0.349069999997963,
                -1.8028741693704615,
                -0.03938029946263444,
                -0.050188152316882007,
                1.3081136206668043,
                -0.44446387143187616,
                -0.10980751225947546,
                -0.2499931438576356,
                0.14999911413368028,
                -0.23342651234973222,
                0.5000280849125882,
                -0.08999917363927584,
                -1.3089000000000002,
                0.044373632637816615,
                -0.2499785604185162,
                -0.15000455806884805,
                -0.08694142021103239,
                0.5000018201989043,
                0.005214674984273897,
                0.13938249352971152,
                0.02751490220235967,
            ],
        },
        {
            "name": "CROUCH",
            "x": [
                -0.3503585862566531,
                -9.108737658717247e-05,
                1.6062449756301738,
                -0.0004949945187189019,
                0.46304714612269776,
                0.0005928818908089678,
                0.4737646949670688,
                -0.038629547533068284,
                -0.03467671549455731,
                -1.6837231635605872,
                0.09428757119986504,
                -0.11810508271720652,
                1.1570348124441838,
                -0.41796138784696785,
                -0.09699440756225168,
                -1.632947593342363,
                -0.019416406861170542,
                0.007292892177137006,
                1.0846878176358126,
                -0.38854248643734046,
                0.07075034855632706,
                -0.24999417749276281,
                0.14999667557432445,
                -0.25763483380664676,
                0.5000291795138719,
                -0.08976327401595742,
                -1.3089000000000002,
                0.04582540894942765,
                -0.24997089900309818,
                -0.14999809992205193,
                -0.08991029362556523,
                0.5000075231878968,
                0.01017607605729893,
                0.16339666524776653,
                0.03656098215447942,
            ],
        },
        {
            "name": "STAND",
            "x": [
                -0.45077635749430106,
                4.881400410751815e-05,
                1.7893411345741133,
                -0.00019376770958076127,
                0.00035731249604661165,
                2.426795731373907e-05,
                0.0011989451794642196,
                -0.00046217245554887917,
                5.6071921122852376e-05,
                -0.1341075908913518,
                0.038651537058946105,
                -0.005303452552842416,
                0.35834865123616383,
                -0.22398820832783678,
                -0.03826564050471655,
                -0.13190274451262213,
                -0.03703041762680692,
                0.004812737564879297,
                0.35417415347606956,
                -0.2224484432867297,
                0.037976426729041474,
                -0.24999829907109655,
                0.14999869320120382,
                -0.23437461859620554,
                0.5000298562538509,
                -0.08929875370726335,
                -1.3089000000195312,
                0.04402943535695793,
                -0.24997177368778528,
                -0.14999792696124484,
                -0.009116742399205655,
                0.5000043213297863,
                0.014892075027384804,
                0.13548541819891394,
                0.0441897172000795,
            ],
        },
    ]
    key = {r["name"]: np.array(r["x"]) for r in rows}
    pre = key["STAND_GROUND"].copy()
    pre[iy] = 0
    nodes = [
        ("READY", pre, 0.6),
        ("GRIPPER_FLIP", key["STAND_GROUND"], 2.0),
        ("PREPARED", key["STAND_GROUND"], 0.5),
    ]
    durations = {
        "WALK_LEFT_CLEAR": 0.65,
        "WALK_LEFT_LAND": 0.65,
        "WALK_RIGHT_CLEAR": 0.65,
        "APPROACH_READY": 0.65,
        "ARM_HOVER": 1.2,
        "FOREARMS": 1.0,
        "RAISE_AND_TUCK": 1.2,
        "RIGHT_KNEE": 1.2,
        "LEFT_KNEE_CLEAR": 1.2,
        "BOTH_KNEES": 1.2,
        "KNEEL_UP": 1.4,
        "RIGHT_FOOT_CLEAR": 1.2,
        "RIGHT_FOOT": 1.0,
        "LEFT_FOOT_CLEAR": 1.2,
        "CROUCH": 1.0,
        "STAND": 1.5,
    }
    for r in rows[1:]:
        nodes.append((r["name"], np.array(r["x"]), durations[r["name"]]))
        if r["name"] == "FOREARMS":
            nodes.append(("FOREARM_HOLD", np.array(r["x"]), 0.65))
    nodes.append(("FINISHED", key["STAND"], 1.0))

    # Preserve the full model states; node features supply continuous Cartesian targets.
    def snapshot(x):
        f = features(x)
        f["arm_points"] = np.array(
            [p.armpts @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b in p.arms]
        )
        f["arm_rot"] = d.xmat[p.arms].reshape(2, 3, 3).copy()
        return f

    def plane_contacts(f):
        foot = {}
        arm = []
        knee = []
        for k, g in enumerate(fg):
            z = f["vertices"][g][:, 2].min()
            loc = (f["feet"][k] - p.box["center"]) @ p.R
            if abs(z - p.floor) < 0.003:
                foot[k] = p.floor
            elif abs(z - p.top) < 0.003 and (abs(loc[:2]) < p.box["half_size"][:2] + 0.03).all():
                foot[k] = p.top
        for k, g in enumerate(p.armgeoms):
            if abs(f["vertices"][g][:, 2].min() - p.top) < 0.003:
                arm.append(k)
        for k in range(2):
            loc = (f["knees"][k] - p.box["center"]) @ p.R
            if abs(kneegap(f, k)) < 0.003 and (abs(loc[:2]) < p.box["half_size"][:2] + 0.03).all():
                knee.append(k)
        return foot, arm, knee

    fps = 15.0
    states = []
    phases = []
    records = []
    previous = None
    for si, (phase, end, duration) in enumerate(nodes):
        start = end if si == 0 else nodes[si - 1][1]
        duration = max(duration, float(np.max(abs(end[6:] - start[6:]))) / 1.4)
        count = max(2, round(duration * fps))
        fa = snapshot(start)
        fb = snapshot(end)
        ca = plane_contacts(fa)
        cb = plane_contacts(fb)
        feetcontact = {k: z for k, z in ca[0].items() if k in cb[0] and abs(cb[0][k] - z) < 0.001}
        armscontact = list(set(ca[1]) & set(cb[1]))
        kneescontact = list(set(ca[2]) & set(cb[2]))
        hold = np.max(abs(end - start)) < 1e-9
        for frame, raw in enumerate(np.linspace(0, 1, count + 1)[1:]):
            u = raw * raw * (3 - 2 * raw)
            ref = (1 - u) * start + u * end
            goal = {
                k: (1 - u) * fa[k] + u * fb[k] for k in ["pelvis", "feet", "knees", "arm_points"]
            }
            feetrot = [
                fa["feetrot"][k]
                @ Rotation.from_rotvec(
                    u * Rotation.from_matrix(fa["feetrot"][k].T @ fb["feetrot"][k]).as_rotvec()
                ).as_matrix()
                for k in range(2)
            ]
            if phase in ["READY", "GRIPPER_FLIP", "PREPARED"] or hold:
                x = ref if phase in ["READY", "GRIPPER_FLIP", "PREPARED"] else previous.copy()
                nfev = 0
            else:

                def residual(x):
                    f = snapshot(x)
                    out = list(180 * (f["pelvis"] - goal["pelvis"]))
                    out.extend((180 * (f["feet"] - goal["feet"])).ravel())
                    out.extend((65 * (f["knees"] - goal["knees"])).ravel())
                    out.extend((120 * (f["arm_points"] - goal["arm_points"])).ravel())
                    out.extend(15 * (x[3:6] - ref[3:6]))
                    out.extend(0.8 * (x[6:] - ref[6:]))
                    for k in range(2):
                        out.extend(
                            12 * Rotation.from_matrix(feetrot[k].T @ f["feetrot"][k]).as_rotvec()
                        )
                    for k, z in feetcontact.items():
                        out.append(9000 * (f["vertices"][fg[k]][:, 2].min() - z + 0.0004))
                    for k in armscontact:
                        out.append(
                            12000 * (f["vertices"][p.armgeoms[k]][:, 2].min() - p.top + 0.0006)
                        )
                    for k in kneescontact:
                        out.append(9000 * (kneegap(f, k) + 0.0004))
                    return np.r_[
                        out,
                        4500 * np.minimum(f["environment"] + 0.0001, 0),
                        2200 * np.minimum(f["selfdist"] + 0.0001, 0),
                        0.3 * (x - (previous if previous is not None else ref)),
                    ]

                seed = previous if previous is not None else ref
                low = lo.copy()
                high = hi.copy()
                if previous is not None:
                    step = np.r_[np.full(3, 0.045), np.full(3, 0.11), np.full(29, 0.17)]
                    low = np.maximum(low, previous - step)
                    high = np.minimum(high, previous + step)
                sol = least_squares(
                    residual,
                    np.clip(seed, low + 1e-8, high - 1e-8),
                    bounds=(low, high),
                    max_nfev=70,
                    diff_step=1e-4,
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                )
                x = sol.x
                nfev = sol.nfev
            previous = x.copy()
            f = snapshot(x)
            states.append(x.copy())
            phases.append(phase)
            r = dict(
                frame=len(states) - 1,
                phase=phase,
                nfev=nfev,
                environment_penetration=float(max(0.0, -f["environment"].min())),
                self_penetration=float(max(0.0, -f["selfdist"].min())),
                pelvis_error=float(np.linalg.norm(f["pelvis"] - goal["pelvis"])),
                expected_foot_planes=feetcontact,
                expected_forearms=armscontact,
                expected_knees=kneescontact,
            )
            records.append(r)
            if frame == count - 1 or len(states) % 20 == 0:
                print(json.dumps(r), flush=True)
        np.savez_compressed(
            (work_dir / "sparse_states.npz"),
            states=np.array(states),
            phases=np.array(phases),
            fps=fps,
        )
        Path((work_dir / "sparse_solver.json")).write_text(json.dumps(records, indent=2))
    print("DONE", len(states), flush=True)


def validate(input_path):
    from check_motion import longest_bout
    import xml.etree.ElementTree as ET

    s = np.load(input_path)
    sparse = s["states"]
    fps = 30.0
    if float(s["fps"]) == 30.0:
        states = sparse.copy()
        phases = s["phase"]
        N = len(states)
    else:
        N = 2 * len(sparse) - 1
        states = np.empty((N, 35))
        states[::2] = sparse
        states[1::2] = (sparse[:-1] + sparse[1:]) / 2
        phases = np.repeat(s["phases"], 2)[:N]
    records = []
    q = []
    roots = []
    quats = []
    coms = []
    pelvis = []
    masks = []
    gripper = m.geom("left_gripper_envelope").id
    regions = [
        "left_ground_foot",
        "right_ground_foot",
        "left_top_foot",
        "right_top_foot",
        "left_forearm",
        "right_forearm",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    ]
    for i, x in enumerate(states):
        f = features(x)
        hits = set()
        events = []
        gripenv = gripself = 0.0
        other = []
        for c in d.contact:
            gs = [int(c.geom1), int(c.geom2)]
            bs = [int(m.geom_bodyid[g]) for g in gs]
            depth = max(0.0, -float(c.dist))
            if gripper in gs:
                if 0 in bs:
                    gripenv = max(gripenv, depth)
                else:
                    gripself = max(gripself, depth)
            if 0 not in bs or c.dist > 0.001:
                continue
            j = 0 if bs[0] else 1
            body = m.body(bs[j]).name
            surface = m.geom(gs[1 - j]).name
            top = surface == "preview_box_0"
            role = None
            if abs(c.frame[2]) < 0.8:
                continue
            for k, side in enumerate(["l", "r"]):
                word = ["left", "right"][k]
                if body == side + "_ankle_x_link":
                    role = word + ("_top_foot" if top else "_ground_foot")
                elif top and body == side + "_wrist_x_link":
                    role = word + "_forearm"
                elif (
                    top
                    and body == side + "_shoulder_z_link"
                    and np.linalg.norm(c.pos - d.xpos[m.body(side + "_elbow_y_link").id]) < 0.105
                ):
                    role = word + "_elbow"
                elif (
                    top
                    and body in [side + "_hip_z_link", side + "_knee_y_link"]
                    and np.linalg.norm(c.pos - f["knees"][k]) < 0.105
                ):
                    role = word + "_knee"
            if role:
                hits.add(role)
            else:
                other.append(body)
            events.append(
                dict(
                    region=role,
                    body=body,
                    surface=surface,
                    point_w=c.pos.tolist(),
                    distance_m=float(c.dist),
                )
            )
        tilt = [float(np.rad2deg(np.arcsin(d.xmat[b].reshape(3, 3)[2, 0]))) for b in p.arms]
        upright = float(
            np.rad2deg(
                np.arccos(np.clip(d.xmat[m.body("l_wrist_z_link").id].reshape(3, 3)[2, 0], -1, 1))
            )
        )
        roottilt = float(np.rad2deg(np.arccos(np.clip(f["rootrot"][2, 2], -1, 1))))
        q.append(d.qpos[p.ids].copy())
        roots.append(d.qpos[:3].copy())
        quats.append(d.qpos[3:7].copy())
        pelvis.append(f["pelvis"])
        coms.append(d.subtree_com[m.body("torso_link").id].copy())
        masks.append([r in hits for r in regions])
        records.append(
            dict(
                frame=i,
                time_s=i / fps,
                phase=str(phases[i]),
                environment_penetration_m=float(max(0.0, -f["environment"].min())),
                non_neighbor_self_penetration_m=float(max(0.0, -f["selfdist"].min())),
                gripper_environment_penetration_m=gripenv,
                gripper_self_penetration_m=gripself,
                contacts=sorted(hits),
                other_contact_bodies=sorted(set(other)),
                events=events,
                forearm_tilt_deg=tilt,
                gripper_angle_to_up_deg=upright,
                torso_tilt_deg=roottilt,
            )
        )
    q = np.array(q)
    roots = np.array(roots)
    quats = np.array(quats)
    masks = np.array(masks)
    pelvis = np.array(pelvis)
    vel = np.vstack([np.diff(q, axis=0) * fps, np.zeros((1, 29))])
    urdf = ET.parse(URDF).getroot()
    limits = np.array(
        [
            [
                float(urdf.find(f'joint[@name="{name}"]/limit').get(k))
                for k in ["lower", "upper", "velocity"]
            ]
            for name in names
        ]
    )
    summary = dict(
        frames=N,
        fps=fps,
        duration_s=(N - 1) / fps,
        joint_limit_violations=int(((q < limits[:, 0] - 1e-5) | (q > limits[:, 1] + 1e-5)).sum()),
        velocity_violations=int((abs(vel) > limits[:, 2] + 1e-5).sum()),
        max_environment_penetration_m=max(r["environment_penetration_m"] for r in records),
        max_non_neighbor_self_penetration_m=max(
            r["non_neighbor_self_penetration_m"] for r in records
        ),
        max_gripper_penetration_m=max(
            max(r["gripper_environment_penetration_m"], r["gripper_self_penetration_m"])
            for r in records
        ),
        longest_dual_forearm_contact_s=longest_bout(masks[:, 4:6].all(1), fps),
        longest_dual_elbow_contact_s=longest_bout(masks[:, 6:8].all(1), fps),
        longest_dual_knee_contact_s=longest_bout(masks[:, 8:10].all(1), fps),
        final_feet_top=bool(masks[-1, 2:4].all()),
        final_torso_tilt_deg=records[-1]["torso_tilt_deg"],
        dynamics_status="NOT RUN",
        training_ready=False,
    )
    phase_summary = []
    for name in dict.fromkeys(phases.tolist()):
        ix = np.flatnonzero(phases == name)
        rr = [records[i] for i in ix]
        phase_summary.append(
            dict(
                phase=name,
                start_s=float(ix[0] / fps),
                end_s=float((ix[-1] + 1) / fps),
                frames=len(ix),
                contact_frames={name: int(masks[ix, k].sum()) for k, name in enumerate(regions)},
                max_environment_penetration_m=max(r["environment_penetration_m"] for r in rr),
                max_self_penetration_m=max(r["non_neighbor_self_penetration_m"] for r in rr),
                gripper_angle_to_up_deg_range=[
                    min(r["gripper_angle_to_up_deg"] for r in rr),
                    max(r["gripper_angle_to_up_deg"] for r in rr),
                ],
            )
        )
    Path((work_dir / "checks.json")).write_text(
        json.dumps(dict(summary=summary, phases=phase_summary, frames=records), indent=2)
    )
    np.savez_compressed(
        (work_dir / "dense_checked.npz"),
        states=states,
        phase=phases,
        joint_pos=q,
        joint_vel=vel,
        base_pos_w=roots,
        base_quat_wxyz=quats,
        pelvis_w=pelvis,
        com_w=np.array(coms),
        geometric_contact_mask=masks,
        contact_region_names=np.array(regions),
    )
    print(json.dumps(summary, indent=2))
    print(json.dumps(phase_summary, indent=2))


def polish():
    f = np.load((work_dir / "dense_checked.npz"))
    states = f["states"].copy()
    phases = f["phase"]
    checks = json.load(open((work_dir / "checks.json")))["frames"]
    original = states.copy()
    records = []
    eg = [m.geom("hiking_" + s + "_shoulder_z_link_0").id for s in ["l", "r"]]
    eb = [m.body(s + "_elbow_y_link").id for s in ["l", "r"]]

    def elbowgap(feat, k):
        v = feat["vertices"][eg[k]]
        mask = np.linalg.norm(v - d.xpos[eb[k]], axis=1) < 0.105
        return float(v[mask, 2].min() - p.top)

    hold = np.flatnonzero(phases == "FOREARM_HOLD")
    hs, he = int(hold[0]), int(hold[-1])
    kneeland = np.flatnonzero(phases == "RIGHT_KNEE")[-1]
    bothland = np.flatnonzero(phases == "BOTH_KNEES")[-1]
    for i, ref in enumerate(original):
        a = np.clip((i - (hs - 12)) / 12, 0, 1) * np.clip((he + 12 - i) / 12, 0, 1)
        a = a * a * (3 - 2 * a)
        need = (
            a > 0
            or checks[i]["environment_penetration_m"] > 0.001
            or checks[i]["non_neighbor_self_penetration_m"] > 0.001
            or kneeland - 5 <= i <= kneeland
            or bothland - 5 <= i <= bothland
        )
        if not need:
            continue
        target = features(ref)
        armpts = np.array([p.armpts @ d.xmat[b].reshape(3, 3).T + d.xpos[b] for b in p.arms])
        g0 = [elbowgap(target, k) for k in range(2)]
        armgap0 = [float(target["vertices"][g][:, 2].min() - p.top) for g in p.armgeoms]
        expected = checks[i]["contacts"]
        feetplanes = {
            k: (p.floor if ["left", "right"][k] + "_ground_foot" in expected else p.top)
            for k in range(2)
            if any(["left", "right"][k] + "_" + v in expected for v in ["ground_foot", "top_foot"])
        }
        activearms = [k for k in range(2) if ["left", "right"][k] + "_forearm" in expected or a > 0]
        activeknees = [k for k in range(2) if ["left", "right"][k] + "_knee" in expected]
        if kneeland - 5 <= i <= kneeland:
            activeknees.append(1)
        if bothland - 5 <= i <= bothland:
            activeknees.extend([0, 1])

        def residual(x):
            cur = features(x)
            out = list(180 * (cur["pelvis"] - target["pelvis"]))
            out.extend((250 * (cur["feet"] - target["feet"])).ravel())
            out.extend((80 * (cur["knees"] - target["knees"])).ravel())
            for k, b in enumerate(p.arms):
                out.extend(
                    (100 * (p.armpts @ d.xmat[b].reshape(3, 3).T + d.xpos[b] - armpts[k])).ravel()
                )
            for k, z in feetplanes.items():
                out.append(10000 * (cur["vertices"][fg[k]][:, 2].min() - z + 0.0004))
            for k in activearms:
                goal = (
                    -0.0005
                    if ["left", "right"][k] + "_forearm" in expected
                    else (1 - a) * armgap0[k] + a * (-0.0005)
                )
                out.append(12000 * (cur["vertices"][p.armgeoms[k]][:, 2].min() - p.top - goal))
            if a > 0:
                for k in range(2):
                    out.append(10000 * (elbowgap(cur, k) - ((1 - a) * g0[k] + a * (-0.0004))))
            for k in set(activeknees):
                out.append(10000 * (kneegap(cur, k) + 0.0004))
            return np.r_[
                out,
                9000 * np.minimum(cur["environment"] + 0.0001, 0),
                7000 * np.minimum(cur["selfdist"] + 0.0001, 0),
                2 * (x - ref),
            ]

        low = np.maximum(lo, ref - np.r_[np.full(3, 0.06), np.full(3, 0.2), np.full(29, 0.25)])
        high = np.minimum(hi, ref + np.r_[np.full(3, 0.06), np.full(3, 0.2), np.full(29, 0.25)])
        sol = least_squares(
            residual,
            np.clip(ref, low + 1e-8, high - 1e-8),
            bounds=(low, high),
            max_nfev=100,
            diff_step=1e-4,
            ftol=1e-7,
            xtol=1e-7,
            gtol=1e-7,
        )
        states[i] = sol.x
        cur = features(sol.x)
        r = dict(
            frame=i,
            phase=str(phases[i]),
            nfev=sol.nfev,
            env=float(max(0, -cur["environment"].min())),
            self=float(max(0, -cur["selfdist"].min())),
            max_joint_correction_rad=float(max(abs(sol.x[6:] - ref[6:]))),
        )
        records.append(r)
        if len(records) % 10 == 0:
            print(json.dumps(r), flush=True)
    np.savez_compressed((work_dir / "polished_states.npz"), states=states, phase=phases, fps=30.0)
    Path((work_dir / "polish_log.json")).write_text(json.dumps(records, indent=2))
    print("DONE", len(records), flush=True)


def export(output):
    import hashlib
    from gripper_geometry import specification

    f = np.load((work_dir / "dense_checked.npz"))
    report = json.load(open((work_dir / "checks.json")))
    out = output
    if out.exists():
        raise FileExistsError(out)
    N = len(f["states"])
    fps = 30.0
    data = {
        k: p.a[k].copy()
        for k in [
            "joint_names",
            "terrain_box_centers_w",
            "terrain_box_quat_wxyz",
            "terrain_box_half_sizes",
        ]
    }
    for k in [
        "phase",
        "joint_pos",
        "joint_vel",
        "base_pos_w",
        "base_quat_wxyz",
        "pelvis_w",
        "com_w",
        "geometric_contact_mask",
        "contact_region_names",
    ]:
        data[k] = f[k].copy()
    data.update(
        fps=np.array(fps),
        timestamps=np.arange(N) / fps,
        qpos=np.c_[f["base_pos_w"], f["base_quat_wxyz"], f["joint_pos"]],
        root_link=np.array("torso_link"),
        source_id=np.array("elf3_full_elbow_climb_demo"),
        motion_origin=np.array("contact_plan"),
        motion_scope=np.array(
            "Full kinematic demo: gripper pre-adjustment, forearm placement, knee transitions, feet on top, stand."
        ),
        support_mode=np.array("forearms_to_knees_to_feet"),
        validation_state=np.array("DEMO_GEOMETRY_CHECKED_DYNAMICS_NOT_RUN"),
        preview_camera_distance=np.array(3.65),
        preview_camera_lookat_w=p.world(p.edge - 0.10, 0, 1.22),
    )
    report["summary"].update(
        platform_width_m=float(p.box["half_size"][1] * 2),
        platform_top_world_z_m=p.top,
        approach_ground_world_z_m=p.floor,
        obstacle_rise_from_approach_m=p.top - p.floor,
        scope="Full geometric demonstration only; no dynamics or force feasibility claim",
        gripper=specification(),
        self_collision_scope="Non-neighbor limb groups via existing checker; not a full within-limb mesh collision audit",
    )
    ph = data["phase"]
    prep = np.isin(ph, ["READY", "GRIPPER_FLIP", "PREPARED"])
    report["summary"]["root_displacement_before_approach_m"] = float(
        np.linalg.norm(f["base_pos_w"][prep] - f["base_pos_w"][0], axis=1).max()
    )
    report["summary"]["left_wrist_y_start_end_flip_deg"] = np.rad2deg(
        f["joint_pos"][
            [0, np.flatnonzero(ph == "GRIPPER_FLIP")[-1]], names.index("l_wrist_y_joint")
        ]
    ).tolist()
    mask = f["geometric_contact_mask"]
    hold = ph == "FOREARM_HOLD"
    report["summary"]["forearm_and_elbow_hold_all_frames"] = bool(mask[hold, 4:8].all())
    report["summary"]["demo_geometry_status"] = (
        "PASS"
        if report["summary"]["joint_limit_violations"] == 0
        and report["summary"]["velocity_violations"] == 0
        and report["summary"]["max_environment_penetration_m"] < 0.001
        and report["summary"]["max_non_neighbor_self_penetration_m"] < 0.001
        and report["summary"]["max_gripper_penetration_m"] < 0.001
        and report["summary"]["forearm_and_elbow_hold_all_frames"]
        and report["summary"]["final_feet_top"]
        else "FAIL"
    )
    report["summary"]["limitations"] = [
        "Geometric playback only; joint torques and contact forces NOT RUN.",
        "Current terrain is the existing widened preview, not a verified 1 m climb.",
        "Gripper is a provisional bounding box; mount and mass are not calibrated.",
        "Forearm hold gripper long axis differs from world vertical by about 15.5 degrees.",
        "Source-inspired contact planning; not an unchanged GMR retarget or accepted RL training reference.",
    ]
    files = [MJCF, URDF, GRIPPER_CONFIG, (OUTPUT / "omniretarget_climb_28_z_scale_0.8.npz")]
    report["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["input_sha256"] = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
    data["contact_plan_metadata_json"] = np.array(json.dumps(report["summary"]))
    np.savez_compressed(out, **data)
    report["summary"]["candidate_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    report["summary"]["file"] = str(out)
    out.with_suffix(".checks.json").write_text(json.dumps(report, indent=2))
    print(out)
    print(json.dumps(report["summary"], indent=2))


if args.reuse_states:
    validate(args.reuse_states)
else:
    build_sequence()
    validate(work_dir / "sparse_states.npz")
    polish()
    validate(work_dir / "polished_states.npz")
export(args.output)
work.cleanup()
