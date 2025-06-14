import taichi as ti
import numpy as np
from Util.TetMesh import TetMesh
from Solver.JacobiXPBD import XPBD_Jacobi_Simulator

import os

ti.init(arch=ti.cuda, device_memory_GB=8)

model_dir = "./Model/MESH"

########################################################################################
####################################### Model path #####################################
########################################################################################



model_names = ["bunny_v534.1.node"]
# keyframePath = "./Scene/target/extern/bunny/dt006/SNH_500_10000/bunny_v534.1_trajectory.pt"
# dt = 0.006

# keyframePath = "./Scene/target/extern/bunny/dt01/SNH_750_1750/bunny_v534.1_trajectory.pt"
# dt = 0.01

keyframePath = "./Scene/target/extern/bunny/dt013/SNH_500_1000/bunny_v534.1_trajectory.pt"
dt = 0.013


end_frame = 40


########################################################################################
###################################### Result path #####################################
########################################################################################

result_path = "./Scene/result/SNH_500_1000_dt013"
# load_path = "C:/Users/media/Desktop/Neural_Constraint_Paper_experiment/Experiment/experiment2/True/SNH_750_1750_dt01_bunny/win_1_2500_relaxation4__15iter_15to5/999999999.pt"
load_path = "./Scene/result/SNH_500_1000_dt013/999999999.pt"

trans_list = [[0.0, 0.0, 0.0]]
scale_list = [1.0]
mesh = TetMesh(model_dir, model_name_list=model_names, trans_list=trans_list,scale_list=scale_list, tetidx_perm=False)

########################################################################################
###################################### ANIMATION #######################################
########################################################################################

anim = False
animation = anim

########################################################################################
####################################### sim param ######################################
########################################################################################

n_substep = 1
num_solver_iter = 15
min_solver_iter = 5

grad_clip = 1.0

relaxation = 4


# energy_model = "corot"
# energy_model = "SNH"
energy_model = "learn"


# mu = 5000.0
# lam = 0.0

mu = 1000.0
lam = 10000.0

young = mu*(3*lam+2*mu)/(lam+mu)
pois = 0.5*lam/(lam+mu)

print("dt : ",dt,"num_iter : ",num_solver_iter)
print("mu= ", mu," lam = ",lam,"E= ", young," pois = ",pois)


########################################################################################
######################################### Task #########################################
########################################################################################

# mode = "mod_stiff_opt"
# mode = "compare_xpbd_pd"

# mode = "param_opt"
mode = "forward_models"


# mode = "export_learned"
# mode = "export_xpbd"

# mode = "export_learned"
# mode = "export_xpbd"
# mode = "export_GT"


method = "step_dual_LNC"

########################################################################################
#################################### training param ####################################
########################################################################################

if method == "step_dual_LNC" :
    lr = 0.005
    lr_min = lr * 0.01
    opt_iter = 1500
    noise_factor = 0
    nn_model_name = 'inv3'
    teacher_forcing = lambda epoch: 1



export_period = 100

sim = XPBD_Jacobi_Simulator(mesh, dt=dt, n_substep=n_substep, relaxation = relaxation, num_solver_iter = num_solver_iter,mu=mu, lam=lam, end_frame=end_frame,
                                   lr=lr, lr_min=lr_min, opt_iter=opt_iter, teacher_forcing=teacher_forcing,grad_clip = grad_clip,
                                method_name= method,energy_model=energy_model,  animation=animation)

assert opt_iter % export_period == 0, "NO!!!!"

def get_result_path(path):
    global export_period, opt_iter
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print("Error: Failed to create the directory.")

    return [path + "/epoch_" + str(i * export_period) + ".pt" for i in range(opt_iter // export_period + 1)]

result_path = get_result_path(result_path)
