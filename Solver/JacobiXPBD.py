
import torch
import numpy as np
import taichi as ti
from tqdm import tqdm
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import os
import math
import time

import Util.teacher_forcing
from Solver.NeuralConstraint import *

import random

seed = 42
torch.manual_seed(seed)  # For PyTorch CPU
torch.cuda.manual_seed_all(seed)  # For PyTorch GPU (if using CUDA)
np.random.seed(seed)  # For NumPy
random.seed(seed)  # For Python's random module

# If using PyTorch's cuDNN, make its behavior deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@ti.data_oriented
class XPBD_Jacobi_Simulator:
    def __init__(self, mesh_dy, dt, n_substep,relaxation,num_solver_iter, mu, lam, end_frame,
                 lr,lr_min,opt_iter,teacher_forcing,grad_clip,method_name,energy_model,animation=False):

        self.loss_each_epoch = []
        cuda = torch.device('cuda')
        self.mesh_dy = mesh_dy
        self.dt = dt
        self.inv_dt = 1/dt
        self.n_substep = n_substep
        self.dt_sub = self.dt / self.n_substep
        self.num_solver_iter = num_solver_iter
        self.num_tet = self.mesh_dy.num_tetras
        self.num_verts = self.mesh_dy.num_verts

        self.endFrame = end_frame

        tet_ids = self.mesh_dy.tet_indices.to_torch().reshape(-1, )

        self.t2v_i = tet_ids[::4].long().cuda()
        self.t2v_j = tet_ids[1::4].long().cuda()
        self.t2v_k = tet_ids[2::4].long().cuda()
        self.t2v_l = tet_ids[3::4].long().cuda()

        self.t2v_ijkl = torch.cat((self.t2v_i, self.t2v_j, self.t2v_k, self.t2v_l), dim=0)

        self.mesh_edge_index = self.get_undirected_edges_from_tets(torch.stack((self.t2v_i, self.t2v_j, self.t2v_k, self.t2v_l), dim=1))
        self.num_edges = self.mesh_edge_index.size()[1]
        self.num_faces = 0
        print("num_verts", self.num_verts, "//", "num_edges", self.num_edges, " //", "num_faces", self.num_faces, " //",
              "num_tets",self.num_tet)

        self.x0 = torch.reshape(self.mesh_dy.x.to_torch(), (-1, 3)).cuda()
        
        self.v0 = torch.zeros_like(self.x0, device=cuda)

        self.x = torch.zeros_like(self.x0, device=cuda)
        self.x_iter = torch.zeros_like(self.x0, device=cuda)
        self.v_iter = torch.zeros_like(self.x0, device=cuda)
        self.v_pred = torch.zeros_like(self.x0, device=cuda)
        self.y = torch.zeros_like(self.x0, device=cuda)
        self.v = torch.zeros_like(self.x, device=cuda)
        self.f_ext = torch.zeros_like(self.x, device=cuda)

        self.dx = torch.zeros_like(self.x0, device=cuda)
        self.jacobi_preconditioned_grad_g = torch.zeros_like(self.x0, device=cuda)

        self.x_traj_gpu = torch.zeros(self.endFrame + 1, self.num_verts, 3).cuda()

        self.x_target_traj_gpu = torch.zeros_like(self.x0).cuda()
        self.y_target_traj_gpu = torch.zeros_like(self.x0).cuda()
        self.v_target_traj_gpu = torch.zeros_like(self.x0).cuda()

        self.g = torch.tensor([0.0, 0.0, 0.0]).repeat(self.num_verts).reshape((-1, 3)).cuda()
        # self.g = torch.tensor([0.0, -9.8, 0.0]).repeat(self.num_verts).reshape((-1, 3)).cuda()

        self.m_inv = torch.reshape(self.mesh_dy.invM.to_torch(), (-1,)).cuda()
        self.m_rep = self.m_inv.reciprocal().repeat_interleave(3).reshape((-1, 3)).cuda()


        self.mass = 0
        if (torch.abs(self.m_inv) < 1e-6).any():
            raise Exception("self.m_inv contains values close to zero")
        else:
            self.mass = self.m_inv.reciprocal().unsqueeze(-1).cuda()
            print(f"self.mass::{self.mass.reciprocal().size()}")

        self.anim_flag = not (not animation)

        if self.anim_flag:
            self.anim = animation
            self.is_fixed1 = torch.from_numpy(self.anim[0]).bool().cuda()
            self.is_fixed2 = torch.from_numpy(self.anim[3]).bool().cuda()
            self.v_l1 = torch.from_numpy(self.anim[1]).cuda()
            self.v_a1 = torch.from_numpy(self.anim[2]).unsqueeze(0).cuda()
            self.v_l2 = torch.from_numpy(self.anim[4]).cuda()
            self.v_a2 = torch.from_numpy(self.anim[5]).unsqueeze(0).cuda()

            self.anim_end_frame = self.anim[-1]

            self.fixed_mask = torch.logical_or(self.is_fixed1, self.is_fixed2)
            self.m_inv[self.fixed_mask] = 0

        self.m_inv_rep = self.m_inv.repeat_interleave(3).reshape((-1, 3)).cuda()
        self.m_inv = self.m_inv.unsqueeze(1)

        self.invDm = self.mesh_dy.invDm.to_torch().cuda()

        self.vol0 = self.mesh_dy.V0.to_torch().cuda()

        self.m_inv_i = self.m_inv[self.t2v_i]
        self.m_inv_j = self.m_inv[self.t2v_j]
        self.m_inv_k = self.m_inv[self.t2v_k]
        self.m_inv_l = self.m_inv[self.t2v_l]

        self.inv_detDm = (1.0 / self.vol0 / 6.0).unsqueeze(-1).unsqueeze(-1)
        self.invDMT = self.invDm.transpose(-2, -1)

        self.mu = torch.tensor([mu], dtype=torch.float32, device='cuda')
        self.lam = torch.tensor([lam], dtype=torch.float32,device='cuda')

        self.k_large = self.mu.clone() if mu > lam else self.lam.clone()

        self.mu_k = (self.mu/self.k_large)
        self.lam_k = (self.lam/self.k_large)

        if energy_model == "corot":
            if lam < 1e-6:
                print("Energy Model :: ARAP", "K_large: ", self.k_large, "mu_k: ", self.mu_k, "lam_k: ", self.lam_k)
                self.get_C_dCdF = self.get_C_dCdx_ARAP
            else :
                print("Energy Model :: COROT", "K_large: ", self.k_large, "mu_k: ", self.mu_k, "lam_k: ", self.lam_k)
                self.get_C_dCdF = self.get_C_dCdx_corot
        elif energy_model == "SNH":
            print("Energy Model :: SNH", "K_large: ", self.k_large, "mu_k: ", self.mu_k, "lam_k: ", self.lam_k)
            self.get_C_dCdF = self.get_C_dCdx_SNH

        elif energy_model == "learn":
            self.get_C_dCdF = self.get_C_dCdx_LNC

        self.inv_compl_Dev = self.mu * self.dt_sub * self.dt_sub * self.vol0
        self.inv_compl_Dev = self.inv_compl_Dev.unsqueeze(-1)
        self.inv_compl_Hyd = self.lam * self.dt_sub * self.dt_sub * self.vol0
        self.inv_compl_Hyd = self.inv_compl_Hyd.unsqueeze(-1)

        self.relaxation = torch.zeros_like(self.m_inv, dtype=torch.int32).cuda()
        self.relaxation[:, :] = relaxation

        self.lr = lr
        self.regularizer = 0.0


        self.Lagrange = torch.zeros_like(self.vol0.unsqueeze(-1), device = cuda)

        self.method_name = method_name
        opt_param = []
        self.nn_model_list = []


        if self.method_name == "step_dual_LNC":
            print("method_name::step_dual_LNC")
            self.nc_FEM = NeuralFEMConstriant_Invariants().cuda()
            self.nn_model_list.append(self.nc_FEM)
            opt_param = [
                {'params': self.nc_FEM.parameters(), 'lr': self.lr},
            ]
            # self.iteration = self.solve_neural_constraint_vel
            self.iteration = self.solve_neural_constraint
        else :
            print(f"[Error] No such method: {method_name}")
            exit(1)


        self.set_models_trainMode = lambda: [m.train() for m in self.nn_model_list]
        self.set_models_evalMode = lambda: [m.eval() for m in self.nn_model_list]

        self.grad_clip = grad_clip
        self.grad_clip_models = lambda:[torch.nn.utils.clip_grad_norm_(m.parameters(),self.grad_clip) for m in self.nn_model_list]


        self.loss_function = self.position_error

        self.opt_iter = opt_iter
        self.lr_min = lr_min
        
        self.optimizer = torch.optim.Adam(opt_param)

        gamma = math.exp(math.log(self.lr_min / self.lr) / (self.opt_iter/2))
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        self.teacher_forcing = teacher_forcing

        self.targetframe = range(1, self.endFrame + 1)

        self.do_model_analysis = False
        self.DO_LEARNING = False


        self.loss_eval = {"pos": [], "F": [], "J": [], "Sig": []}
        self.invariants_hist_data = torch.zeros(end_frame, self.num_tet, 5).cuda()
        self.C_gt_hist_data = torch.zeros(end_frame, self.num_tet, 1).cuda()

        ################################# Render
        self.pos_sequence_ti = ti.Vector.field(3, ti.f32, shape=(self.endFrame + 1, self.num_verts))
        self.pos = ti.Vector.field(3, ti.f32, (self.num_verts,))

        self.pos_sequence_target_ti = ti.Vector.field(3, ti.f32, shape=(self.endFrame + 1, self.num_verts))
        self.pos_target = ti.Vector.field(3, ti.f32, (self.num_verts,))

        self.frame_ti = ti.field(ti.i32, shape=())

        self.pos_seq_cursor = 0
        self.list_pos_sequence_path = []
        self.num_pos_sequence = 0

        # def get_available_filename(base_name):
        #     if not os.path.exists(base_name):
        #         return base_name
        # 
        #     name, ext = os.path.splitext(base_name)
        #     i = 1
        #     while True:
        #         candidate = f"{name}_{i}{ext}"
        #         if not os.path.exists(candidate):
        #             return candidate
        #         i += 1
        #
        # filename = get_available_filename("invDm.pt")
        # torch.save(self.invDm, filename)
        # print(f"Saved to: {filename}")

        plt.ion()
 
    def init_forward(self):
        with torch.no_grad():
            self.x = self.x_target_traj_gpu[0].clone()
            self.v = self.v0.clone()
            self.x_traj_gpu[0] = self.x.clone()

        with torch.no_grad():
            if torch.norm(self.x_traj_gpu[0] - self.x_target_traj_gpu[0]) > 1e-3:
                print(self.x_traj_gpu[0])
                print(self.x_target_traj_gpu[0])
                print(torch.norm(self.x_traj_gpu[0] - self.x_target_traj_gpu[0]))
                raise "ERROR::SIM Initial state is different to Initial state of Target trajectory!!!!"

    def prediction_dual(self):
        self.y = self.x + self.dt_sub * (self.v + self.dt_sub * self.m_inv_rep * self.f_ext)

    def solve_neural_constraint_vel(self):
        x_iter = self.x + self.dt_sub * self.v_iter
        x_iter_stack = torch.stack((x_iter[self.t2v_i], x_iter[self.t2v_j], x_iter[self.t2v_k]), dim=2)
        Ds = x_iter_stack - x_iter[self.t2v_l].unsqueeze(2)
        F = Ds @ self.invDm

        C, dCdF = self.get_C_dCdx_LNC(F)
        gradC_x = dCdF @ self.invDMT

        grad_vi = self.dt * gradC_x[:, :, 0]
        grad_vj = self.dt * gradC_x[:, :, 1]
        grad_vk = self.dt * gradC_x[:, :, 2]
        grad_vl = -(grad_vi + grad_vj + grad_vk)

        norm_sqr_grad_vi = torch.sum(grad_vi ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vj = torch.sum(grad_vj ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vk = torch.sum(grad_vk ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vl = torch.sum(grad_vl ** 2, dim=1, keepdim=True)

        schur = self.m_inv_i * norm_sqr_grad_vi + self.m_inv_j * norm_sqr_grad_vj + self.m_inv_k * norm_sqr_grad_vk + self.m_inv_l * norm_sqr_grad_vl

        alpha = 1.0 / self.vol0.unsqueeze(-1)
        ld = - (C + alpha * self.Lagrange) / (schur + alpha)

        self.Lagrange = self.Lagrange + ld

        dv_i = self.m_inv_i * ld * grad_vi
        dv_j = self.m_inv_j * ld * grad_vj
        dv_k = self.m_inv_k * ld * grad_vk
        dv_l = self.m_inv_l * ld * grad_vl

        dv_ijkl = torch.cat((dv_i, dv_j, dv_k, dv_l), dim=0)
        self.dv = torch.zeros_like(self.v_iter, device='cuda').index_add(0, self.t2v_ijkl, dv_ijkl)

        self.v_iter = self.v_iter + self.dv / self.relaxation

    def solve_neural_constraint(self):
        xli = self.y[self.t2v_i] - self.y[self.t2v_l]
        xlj = self.y[self.t2v_j] - self.y[self.t2v_l]
        xlk = self.y[self.t2v_k] - self.y[self.t2v_l]

        Ds = torch.stack((xli, xlj, xlk), dim=2)
        F = Ds @ self.invDm

        C, dCdF = self.get_C_dCdx_LNC(F)
        gradC_x = dCdF @ self.invDMT

        grad_xi = gradC_x[:, :, 0]
        grad_xj = gradC_x[:, :, 1]
        grad_xk = gradC_x[:, :, 2]
        grad_xl = -(grad_xi + grad_xj + grad_xk)

        norm_sqr_grad_i = torch.sum(grad_xi ** 2, dim=1, keepdim=True)
        norm_sqr_grad_j = torch.sum(grad_xj ** 2, dim=1, keepdim=True)
        norm_sqr_grad_k = torch.sum(grad_xk ** 2, dim=1, keepdim=True)
        norm_sqr_grad_l = torch.sum(grad_xl ** 2, dim=1, keepdim=True)

        schur = self.m_inv_i * norm_sqr_grad_i + self.m_inv_j * norm_sqr_grad_j + self.m_inv_k * norm_sqr_grad_k + self.m_inv_l * norm_sqr_grad_l

        alpha = 1.0 / self.vol0.unsqueeze(-1)
        ld = - (self.dt_sub ** 2) * (C + alpha * self.Lagrange) / ((self.dt_sub ** 2) * schur + alpha)

        self.Lagrange = self.Lagrange + ld

        dx_i = self.m_inv_i * ld * grad_xi
        dx_j = self.m_inv_j * ld * grad_xj
        dx_k = self.m_inv_k * ld * grad_xk
        dx_l = self.m_inv_l * ld * grad_xl

        dx_ijkl = torch.cat((dx_i, dx_j, dx_k, dx_l), dim=0)
        self.dx = torch.zeros_like(self.dx, device='cuda').index_add(0, self.t2v_ijkl, dx_ijkl)

        self.y = self.y + self.dx / self.relaxation

    def step_learning_neural_constraint_procedural(self,frame):
        self.move_fixed_point(frame)
        self.f_ext = self.m_rep * self.g

        self.Lagrange = torch.zeros_like(self.Lagrange)

        self.y = self.x + self.dt * (self.v + self.dt * self.m_inv_rep * self.f_ext)
        self.v_pred = self.v + self.dt * self.m_inv_rep * self.f_ext

        num_iter = self.iter_list[frame]

        for st in range(num_iter):
            self.iteration()

        #### Update
        self.v = (self.y - self.x) / self.dt
        self.x = self.y

    def step_learning_neural_constraint(self,frame):
        self.move_fixed_point(frame)
        self.f_ext = self.m_rep * self.g

        self.Lagrange = torch.zeros_like(self.Lagrange)

        self.y = self.x + self.dt * (self.v + self.dt * self.m_inv_rep * self.f_ext)
        self.v_pred = self.v + self.dt * self.m_inv_rep * self.f_ext

        for st in range(self.num_solver_iter):
            self.iteration()

        #### Update
        self.v = (self.y - self.x) / self.dt
        self.x = self.y

    def step_learning_neural_constraint_procedural_vel(self,frame):
        self.move_fixed_point(frame)
        self.f_ext = self.m_rep * self.g

        self.Lagrange = torch.zeros_like(self.Lagrange)

        self.v_pred = self.v + self.dt * self.m_inv_rep * self.f_ext
        self.v_iter = self.v_pred

        num_iter = self.iter_list[frame]
        #### Projection
        for st in range(num_iter):
            self.iteration()

        #### Update
        self.v = self.v_iter
        self.x = self.x + self.dt * self.v




    def floor_collision_force_add(self,):

        floor_y = -2.0
        pos_y = self.x[:, 1]
        pos_mask = pos_y < floor_y
        force_magnitude = torch.zeros_like(pos_y)
        force_magnitude[pos_mask] -= 100.0 / 15.0 * (pos_y[pos_mask] - (floor_y))
        # force_magnitude[pos_mask] -= 100.0 * (pos_y[pos_mask] - (- 3.0))
        force = torch.zeros_like(self.x)  # (662, 3)
        force[:, 1] = force_magnitude
        self.f_ext += force


    def floor_001collision_force_add(self,):

        floor_y = -1.2
        pos_y = self.x[:, 1]
        pos_y_mask = pos_y < floor_y

        force_y = torch.zeros_like(pos_y)
        force_y[pos_y_mask] = -100.0  * (pos_y[pos_y_mask] - (floor_y))

        mu = 0.2
        force_friction = mu * force_y.abs()

        v_xz = self.v[:, [0, 2]]
        v_xz_norm = torch.norm(v_xz, dim=1) + 1e-12

        fric_dy_mask = (v_xz_norm > force_friction) & pos_y_mask  # [n]
        decay = max(1.0 - 0.1, 0.0)

        self.v[fric_dy_mask, 0] *= decay
        self.v[fric_dy_mask, 2] *= decay

        fric_st_mask = (v_xz_norm <= force_friction) & pos_y_mask
        self.v[fric_st_mask, 0] = 0.0
        self.v[fric_st_mask, 2] = 0.0


        force_magnitude_x = torch.zeros_like(pos_y)
        pos_x = self.x[:, 0]
        pos_x_mask = pos_x < -3.0
        force_magnitude_x[pos_x_mask] -= 100.0/4.0  * (pos_x[pos_x_mask] - (- 3.0))

        force = torch.zeros_like(self.x)  # [n, 3]
        # force[:, 1] = force_y
        force[:, 0] = force_magnitude_x

        self.f_ext += force






    def solve_stiffness_optimization(self, ):

        #################################################### projection
        x_iter = self.x + self.dt * self.v_iter
        x_iter_stack = torch.stack((x_iter[self.t2v_i], x_iter[self.t2v_j], x_iter[self.t2v_k]), dim=2)
        Ds = x_iter_stack - x_iter[self.t2v_l].unsqueeze(2)
        F = Ds @ self.invDm
        C,dCdF = self.get_C_dCdF(F)
        gradC_x = dCdF @ self.invDMT

        grad_vi = self.dt * gradC_x[:, :, 0]
        grad_vj = self.dt * gradC_x[:, :, 1]
        grad_vk = self.dt * gradC_x[:, :, 2]
        grad_vl = -(grad_vi + grad_vj + grad_vk)

        norm_sqr_grad_vi = torch.sum(grad_vi ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vj = torch.sum(grad_vj ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vk = torch.sum(grad_vk ** 2, dim=1, keepdim=True)
        norm_sqr_grad_vl = torch.sum(grad_vl ** 2, dim=1, keepdim=True)

        schur = self.m_inv_i * norm_sqr_grad_vi + self.m_inv_j * norm_sqr_grad_vj + self.m_inv_k * norm_sqr_grad_vk + self.m_inv_l * norm_sqr_grad_vl

        mu_cor = self.k_large * self.k_opt_mul
        alpha = 1.0 / mu_cor / self.vol0.unsqueeze(-1)
        ld = - (C + alpha * self.Lagrange) / (schur + alpha)
        # ld = - (C_Dev) / (schur + 1.0 / self.mu / self.vol0.unsqueeze(-1) )

        self.Lagrange = self.Lagrange + ld

        dv_i = self.m_inv_i * ld * grad_vi
        dv_j = self.m_inv_j * ld * grad_vj
        dv_k = self.m_inv_k * ld * grad_vk
        dv_l = self.m_inv_l * ld * grad_vl

        dv_ijkl = torch.cat((dv_i, dv_j, dv_k, dv_l), dim=0)
        self.dv = torch.zeros_like(self.v_iter, device='cuda').index_add(0, self.t2v_ijkl, dv_ijkl)

        self.v_iter = self.v_iter + self.dv / self.relaxation


    def get_C_dCdx(self,F,E_model):

        C = None
        gradC_x = None

        if E_model == "ARAP":
            # R = self.get_R_diff(F)
            R = self.get_R(F)

            F_R = F-R
            C = torch.norm(F_R,dim=(1,2)).unsqueeze(-1)

            gradC_x = ((F_R) @ self.invDMT) / (C.unsqueeze(-1) + 1e-6)

        return C,gradC_x

    def get_undirected_edges_from_tets(self, tets):

        i, j, k, l = tets[:, 0], tets[:, 1], tets[:, 2], tets[:, 3]
        edges = torch.cat([
            torch.stack([i, j], dim=1),
            torch.stack([i, k], dim=1),
            torch.stack([i, l], dim=1),
            torch.stack([j, k], dim=1),
            torch.stack([j, l], dim=1),
            torch.stack([k, l], dim=1),
        ], dim=0)

        edges = torch.sort(edges, dim=1)[0]

        edges = torch.unique(edges, dim=0)

        edge_index = edges.t()

        return edge_index

    def get_C_dCdx_ARAP(self, F):
        R = self.get_R_diff(F)
        F_R = F - R
        C = torch.norm(F_R, dim=(1, 2)).unsqueeze(-1)
        dCdF = (F_R) / (C.unsqueeze(-1) + 1e-6)

        return 1.41421* self.mu_k.sqrt() * C, 1.41421* self.mu_k.sqrt() * dCdF

    def get_C_dCdx_corot(self, F):
        print("not_impl")
        exit()
        R = self.get_R_diff(F)
        F_R = F - R
        F_R_norm = torch.norm(F_R, dim=(1, 2)).unsqueeze(-1)

        Sig = torch.linalg.svdvals(F)
        I1 = torch.sum(Sig, dim=1, keepdim=True)

        C_sqr = self.mu_k * F_R_norm**2 + self.lam_k * (I1- 3)**2
        C = C_sqr.sqrt()

        dC_sqrdF = 2 * (self.mu_k * F_R_norm * (F_R) + self.lam_k * (I1- 3) * R)
        dCdF = 0.5 * dC_sqrdF / (C.unsqueeze(-1) + 1e-5)

        return C,dCdF

    def get_C_dCdx_SNH(self,F, impl_autodiff = False):
        if impl_autodiff :
            with torch.enable_grad():
                F.requires_grad_(True)

                I2 = torch.sum(F ** 2, dim=(1, 2)).unsqueeze(-1)
                I3 = torch.det(F).unsqueeze(-1)

                C_sqr = self.mu_k * (I2 - 3) - 2 * self.mu_k * (I3 - 1) + self.lam_k * (I3 - 1) * (I3 - 1)  # E_snh = 1/2 k_large C^2
                C = C_sqr.sqrt()
                dCdF = torch.autograd.grad(C.sum(), F, create_graph=self.DO_LEARNING)[0]

                if not self.DO_LEARNING:
                    F.detach_()
        else :
            I2 = torch.sum(F ** 2, dim=(1, 2)).unsqueeze(-1)
            I3 = torch.det(F).unsqueeze(-1)

            f1 = F[:, :, 0]
            f2 = F[:, :, 1]
            f3 = F[:, :, 2]
            P1 = torch.cross(f2, f3, dim=1)
            P2 = torch.cross(f3, f1, dim=1)
            P3 = torch.cross(f1, f2, dim=1)
            dI3dF = torch.stack((P1, P2, P3), dim=2)

            C_sqr = self.mu_k * (I2 - 3) - 2 * self.mu_k * (I3 - 1) + self.lam_k * (I3 - 1) * (I3 - 1)  # 2 * snh
            C = C_sqr.sqrt()

            dC_sqrdF = 2 * (self.mu_k * F - self.mu_k * dI3dF + self.lam_k * (I3.unsqueeze((-1)) - 1) * dI3dF)

            dCdF = 0.5*dC_sqrdF/(C.unsqueeze(-1) + 1e-5)

        return C,dCdF

    def get_C_dCdx_LNC(self, F):

        inv1, inv2, inv3, ginv1, ginv2, ginv3 = self.get_CG_invariants(F,invset = 2)

        with torch.enable_grad():
            invariants = torch.cat((inv1, inv2, inv3), dim=1).requires_grad_(True)

            C_FEM = self.nc_FEM(invariants)
            C_FEM_sum = C_FEM.sum()

            gradC_rinv123 = torch.autograd.grad(C_FEM_sum, invariants, create_graph=self.DO_LEARNING)[0]
            gradC_rinv123_unsqz = gradC_rinv123.unsqueeze(-1)
            if not self.DO_LEARNING:
                invariants.detach_()

        dCdF = gradC_rinv123_unsqz[:, :1, :] * ginv1 + gradC_rinv123_unsqz[:, 1:2, :] * ginv2 + gradC_rinv123_unsqz[
                                                                                                   :, 2:3, :] * ginv3
        return C_FEM, dCdF

    def get_CG_invariants(self, F, invset, differentiable = True):

        FTF = F.transpose(-2, -1) @ F
        J = torch.det(F).unsqueeze(-1)
        if invset == 2:

            Sig = torch.linalg.svdvals(F)
            tr = torch.sum(Sig, dim=1, keepdim=True)
            Ic = torch.sum(F ** 2, dim=(1, 2)).unsqueeze(-1)

            CG_inv1_normalized = tr - 3
            CG_inv2_normalized = Ic - 3
            CG_inv3_normalized = J - 1

            gI1 = 2 * F
            gI2 = 4 * (F @ (FTF))

            deriv_inv2_F = gI1

            f1 = F[:, :, 0]
            f2 = F[:, :, 1]
            f3 = F[:, :, 2]
            P1 = torch.cross(f2, f3, dim=1)
            P2 = torch.cross(f3, f1, dim=1)
            P3 = torch.cross(f1, f2, dim=1)
            gJ = torch.stack((P1, P2, P3), dim=2)

            deriv_inv3_F = gJ

            const = 4 * (tr ** 3) - (4 * Ic * tr) - (8 * J)

            with torch.no_grad():
                const_mask = torch.abs(const).squeeze(-1) < 1e-6

            const[const_mask, :] = 1e-6

            gtr1 = (2 * tr * tr + 2 * Ic) / const
            gtr2 = -2 / const
            gtrJ = (8 * tr) / const

            if differentiable :
                R = (gtr1.unsqueeze(-1) * gI1 + gtr2.unsqueeze(-1) * gI2 + gtrJ.unsqueeze(-1) * gJ)

                R[const_mask, :, :] = torch.eye(3, device='cuda')

                # with torch.no_grad():
                #     detchk = torch.linalg.det(R)
                #     tol = 1e-3
                #     inval = torch.abs(detchk - 1.0) > tol
                #     num_invalid = inval.sum().item()
                #     if num_invalid !=0 :
                #         print("DEBUG::umumumumumumumum : ",num_invalid)

                deriv_inv1_F = R
            else :
                deriv_inv1_F = self.get_R(F)

            return CG_inv1_normalized, CG_inv2_normalized, CG_inv3_normalized, deriv_inv1_F, deriv_inv2_F, deriv_inv3_F
        else :
            print("get_CG_invariants:: not implemented version")



    def update_dual(self):
        self.y = self.y + self.dx / self.relaxation
        self.v = (self.y - self.x) / self.dt
        self.x = self.y


    def move_fixed_point(self,cur_frame):
        if self.anim_flag and cur_frame < self.anim_end_frame:
            c1 = torch.mean(self.x[self.is_fixed1],dim=0)
            c2 = torch.mean(self.x[self.is_fixed2],dim=0)

            displ1 = (self.x[self.is_fixed1] - c1)
            displ2 = (self.x[self.is_fixed2] - c2)

            v_a1 = torch.cross(self.v_a1,displ1,dim=1)
            v_a2 = torch.cross(self.v_a2,displ2,dim=1)

            self.x[self.is_fixed1] = self.x[self.is_fixed1] + (self.v_l1 + v_a1) * self.dt
            self.x[self.is_fixed2] = self.x[self.is_fixed2] + (self.v_l2 + v_a2) * self.dt



    def step_stiff(self,frame):
        self.move_fixed_point(frame)
        self.f_ext = self.m_rep * self.g
        self.Lagrange = torch.zeros_like(self.Lagrange)

        #pred
        self.v_pred = self.v + self.dt_sub * self.m_inv_rep * self.f_ext
        self.v_iter = self.v_pred

        #### Proj
        for st in range(self.num_solver_iter):
            self.solve_stiffness_optimization()

        #### Update
        self.v = self.v_iter
        self.x = self.x + self.dt * self.v


    def do_sim(self, path, gen_keyframe=False):
        self.DO_LEARNING = False

        self.k_opt_mul = torch.tensor([1], dtype=torch.float32, device='cuda')

        losses_pos = []
        target_counter = 0

        with torch.no_grad():
            self.init_forward()
            for nt in tqdm(range(self.endFrame)):
                self.step_stiff(nt)
                self.x_traj_gpu[nt + 1] = self.x

                if nt + 1 in self.targetframe:
                    loss_pos = ((self.x - self.x_target_traj_gpu[self.targetframe[target_counter]]) ** 2).sum(dim=1).mean()
                    losses_pos.append(loss_pos.item())
                    target_counter = target_counter + 1
        
        print(sum(losses_pos))

        self.pos_sequence_torch_serial = self.x_traj_gpu.reshape((-1,))

        self.set_pos_sequence()

    def do_sim_evaluation(self, path, gen_keyframe=False):
        self.DO_LEARNING = False

        self.set_models_evalMode()

        losses_pos = []
        target_counter = 0

        losses_gx = []

        with torch.no_grad():
            self.init_forward()
            for nt in tqdm(range(self.endFrame)):
                self.step_learning_neural_constraint(nt)
                self.x_traj_gpu[nt + 1] = self.x

                # cur_linmom = self.mass * self.v
                # print(cur_linmom.sum(dim=(0)))

                if nt + 1 in self.targetframe:
                    loss_pos = ((self.x - self.x_target_traj_gpu[self.targetframe[target_counter]]) ** 2).sum(dim=1).mean()
                    losses_pos.append(loss_pos.item())

                    gx = self.physics_informed_energy(self.v,self.v_pred)
                    losses_gx.append(gx.item())
                    target_counter = target_counter + 1

                self.x_traj_gpu[nt + 1] = self.x

        print(sum(losses_pos))

        print(losses_gx)


        self.pos_sequence_torch_serial = self.x_traj_gpu.reshape((-1,))
        self.set_pos_sequence()

    def do_sim_generalization_task(self, path, gen_keyframe=False):
        self.DO_LEARNING = False

        self.set_models_evalMode()

        self.x_traj_gpu = torch.zeros(self.endFrame + 1, self.num_verts, 3).cuda()

        with torch.no_grad():
            self.init_largedt_show()

            for nt in tqdm(range(self.endFrame)):
                self.step_learning_neural_constraint_generalization_task_largedt_PBD(nt)


                self.x_traj_gpu[nt + 1] = self.x

        self.pos_sequence_torch_serial = self.x_traj_gpu.reshape((-1,))
        self.set_pos_sequence()

    def export_traj(self, res_path):
        torch.save(self.x_traj_gpu, res_path)

        # print("Do not Export", res_path)
        self.num_pos_sequence = self.num_pos_sequence + 1
    def export_loss_each_epoch (self, res_path, max_iter, export_period):
        num_loss = len(self.loss_each_epoch)
        xs = [x for x in range(num_loss)]
        plt.plot(xs, self.loss_each_epoch, '-o')
        plt.xticks(range(0, max_iter+1, export_period))

        if num_loss > 15:
            recent_losses = self.loss_each_epoch[-15:]
        else:
            recent_losses = self.loss_each_epoch

        upper_bound = max(recent_losses) * 1.5  # Adding a bit of padding to the max value

        # dynamically set axis upperbound
        plt.ylim(-1e-4,upper_bound)

        # Set the y-axis limits
        if math.isnan(upper_bound) :
            print("NAN EXIST!!!")
            exit()

        plt.savefig(os.path.dirname(res_path) + '/loss_each_epoch_plot.png', format='png')
        plt.draw()
        plt.pause(0.001)
        plt.clf()

        with open(os.path.dirname(res_path) +"/loss_each_epoch.txt", "w") as output:
            for value in self.loss_each_epoch:
                output.write(str(value) + "\n")
    def export_loss_eval(self, res_path, max_iter, export_period):

        for measure in self.loss_eval.keys():

            num_loss = len(self.loss_eval[measure])
            xs = [x * export_period for x in range(num_loss)]

            plt.plot(xs, self.loss_eval[measure], '-o')
            plt.title(measure+"_loss")

            plt.xticks(range(0, max_iter+1))
            plt.savefig(os.path.dirname(res_path) + '/loss_eval_plot_' + measure + '.png', format='png')
            plt.clf()

            with open(os.path.dirname(res_path) +"/loss_eval" + measure + ".txt", "w") as output:
                for value in self.loss_eval[measure]:
                    output.write(str(value) + "\n")

    def make_checkpoint(self,EPOCH,LOSS,PATH):

        if self.method_name in ["step_dual_LNC","step_dual_LNC_adjust"] :
            PATH = os.path.dirname(PATH) + '/nc_FEM' + '_' + str(EPOCH) + '.pt'
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': self.nc_FEM.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': LOSS,
            }, PATH)

        if self.method_name in ["step_dual_LNC_adjust", "step_dual_Adjust"] :
            PATH = os.path.dirname(PATH) + '/adjustment' + '_' + str(EPOCH) + '.pt'
            torch.save({
                'epoch': EPOCH,
                'model_state_dict': self.adjustment.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': LOSS,
            }, PATH)

    def load_checkpoint(self,PATH):

        numdotpt = os.path.basename(PATH)
        DIRNAME = os.path.dirname(PATH)
        if self.method_name in ["step_dual_LNC","step_dual_LNC_adjust"] :
            NCPATH = DIRNAME + "/nc_FEM" + '_' + numdotpt
            self.nc_FEM.load_state_dict(torch.load(NCPATH)['model_state_dict'])
            self.nc_FEM.eval()

        if self.method_name in ["step_dual_LNC_adjust", "step_dual_Adjust"] :
            ADJPATH= DIRNAME + "/adjustment" + '_' + numdotpt
            self.adjustment.load_state_dict(torch.load(ADJPATH)['model_state_dict'])
            self.adjustment.eval()
            
            
    def load_target(self, path):

        self.x_target_traj_gpu = torch.load(path).cuda()  # size() = [# of time step, # of vertex, # of dimension]
        print("==========================")

        last_frame = self.x_target_traj_gpu.size()[0] - 1

        if last_frame > self.endFrame:
            last_frame = self.endFrame
            self.x_target_traj_gpu = self.x_target_traj_gpu[:self.endFrame+1,:,:]
            print(self.x_target_traj_gpu.size())

        print("LOAD_TARGET", self.x_target_traj_gpu.device, self.x_target_traj_gpu.size())
        x_target_torch_cpu = self.x_target_traj_gpu.reshape((self.endFrame + 1, self.num_verts, 3)).cpu()

        pos_sequence_target_np = np.array(x_target_torch_cpu.detach())
        self.pos_sequence_target_ti.from_numpy(pos_sequence_target_np)
        self.pos_target.from_numpy(pos_sequence_target_np[last_frame, :, :])

        v_path = path.replace(".pt", "_v.pt")
        self.v_target_traj_gpu = torch.load(v_path).cuda()  # size() = [# of time step, # of vertex, # of dimension]
        self.v_target_traj_gpu = self.v_target_traj_gpu[:self.endFrame + 1, :, :]
        print("LOAD_TARGET_v", self.v_target_traj_gpu.device)

        y_path = path.replace(".pt", "_Y.pt")
        if os.path.exists(y_path):
            self.y_target_traj_gpu = torch.load(y_path).cuda()  # size() = [# of time step, # of vertex, # of dimension]
            print("LOAD_TARGET_Y", self.y_target_traj_gpu.device)
        print("==========================")


    def load_render_pos_seq(self, path_list):
        self.list_pos_sequence_path = path_list
        self.num_pos_sequence = len(path_list)

        print("========", self.num_pos_sequence, "sequences are loaded!!! ========")

        pos_sequence_torch_serial = torch.load(path_list[0]).reshape((-1,)).cpu()
        pos_sequence_torch = torch.reshape(pos_sequence_torch_serial,(self.endFrame+1,self.num_verts,3))

        self.pos_sequence_ti.from_numpy(np.array(pos_sequence_torch.detach()))

    def load_next_pos_seq(self,ISUP =True):

        if ISUP:
            print("pos_sequence",(self.pos_seq_cursor+1) % self.num_pos_sequence, " / ", self.num_pos_sequence)
            self.pos_seq_cursor = (self.pos_seq_cursor+1) % self.num_pos_sequence
        else :
            print("pos_sequence",(self.pos_seq_cursor-1) % self.num_pos_sequence, " / ", self.num_pos_sequence)
            self.pos_seq_cursor = (self.pos_seq_cursor-1) % self.num_pos_sequence

        pos_sequence_torch_serial = torch.load(self.list_pos_sequence_path[self.pos_seq_cursor]).cpu()
        pos_sequence_torch = torch.reshape(pos_sequence_torch_serial,(self.endFrame+1,self.num_verts,3))
        self.pos_sequence_ti.from_numpy(np.array(pos_sequence_torch.detach()))

    def set_pos_sequence(self):
        pos_sequence_torch_cpu = torch.reshape(self.pos_sequence_torch_serial,(self.endFrame+1,self.num_verts,3)).cpu()

        pos_sequence_np = np.array(pos_sequence_torch_cpu.detach())
        self.pos_sequence_ti.from_numpy(pos_sequence_np)

    @ti.kernel
    def set_render_pos(self,f:int,target_dy:int,offset_flag:int):
        for v in self.pos :
            self.pos[v] = self.pos_sequence_ti[f,v]
            if target_dy == 1 or 2:

                offset = ti.Vector([0,0,0])
                if offset_flag :
                    offset = ti.Vector([3,0,0])

                self.pos_target[v] = self.pos_sequence_target_ti[f, v] + offset

            # elif target_dy == 2 :
            #     self.pos_target[v] = self.pos_sequence_target_ti[self.endFrame, v]
            # if v==0 or v==1 or v==2 or v==3  :
            #     print(v, self.pos[v])

    def get_angular_momemtum(self,xt,pt):
        
        each_element_angular_momentum = torch.linalg.cross(xt, pt)

        tot_angular_momentum = each_element_angular_momentum.sum(dim=0)
        return tot_angular_momentum

    def LOG_GT_momentum(self):
        all_mass = self.mass.unsqueeze(0)
        linear_momentum = (all_mass * self.v_target_traj_gpu)
        all_linear_momentum = linear_momentum.sum(dim=(1))
        print("Linear Momentum at Last Frame",all_linear_momentum[-1,:])
        all_angular_momentum = torch.zeros_like(all_linear_momentum)

        for t in range(self.endFrame+1) :

            all_angular_momentum[t] = self.get_angular_momemtum( self.x_target_traj_gpu[t,:,:] ,linear_momentum[t,:,:] )

            # if t == self.endFrame :
            #     print("Angular Momentum at Last Frame",all_angular_momentum[t])
            print("Angular Momentum at Last Frame",all_angular_momentum[t])

        return all_linear_momentum,all_angular_momentum

    def hist_GT_velocity(self):
        vecs = self.v_target_traj_gpu.reshape(-1, 3).clone()
        norms = torch.norm(vecs, dim=1)

        norms_np = norms.cpu().numpy()

        num_bins = 10
        hist, bin_edges = np.histogram(norms_np, bins=num_bins)

        for i in range(len(hist)):
            print(f"[{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}): {hist[i]}")
    def physics_informed_energy(self,v,v_pred):

        v_norm_sqr = torch.sum((v-v_pred) ** 2,dim=1,keepdim=True)
        E_inertial = (0.5 * self.mass * v_norm_sqr).sum()

        x_iter = self.x
        x_iter_stack = torch.stack((x_iter[self.t2v_i], x_iter[self.t2v_j], x_iter[self.t2v_k]), dim=2)
        Ds = x_iter_stack - x_iter[self.t2v_l].unsqueeze(2)
        F = Ds @ self.invDm

        inv1, inv2, inv3, _,__,___= self.get_CG_invariants(F,invset = 2)
        invariants = torch.cat((inv1, inv2, inv3), dim=1)
        C_FEM = self.nc_FEM(invariants)
        
        E_elastic = (0.5 * self.vol0.unsqueeze(-1) *(C_FEM**2) ).sum()

        return E_inertial + E_elastic

    def param_optimize_TBPTT(self, result_path, export_period=10):

        max_iter = self.opt_iter
        teacher_forcing = self.teacher_forcing

        self.DO_LEARNING = True
        start_param = time.time()
        print("====== OPTIMIZE PARAMETER======")

        self.hist_GT_velocity()

        for epoch in range(max_iter):
            self.set_models_trainMode()

            start_iter_epoch = time.time()
            losses = []

            window_size = teacher_forcing(epoch)

            start_frame_last_seq = self.endFrame - window_size
            start_frame_idx = list(range(start_frame_last_seq + 1))
            random.shuffle(start_frame_idx)

            print("winsize", window_size)

            for start_frame in start_frame_idx:

                traj_loss = 0.0
                validity = True

                if start_frame == 0:
                    self.init_forward()
                else:
                    self.x = self.x_target_traj_gpu[start_frame].clone()
                    self.v = self.v_target_traj_gpu[start_frame].clone()

                for nt in range(start_frame, start_frame + window_size):
                    self.step_learning_neural_constraint(nt)

                    # loss = self.loss_function(self.v, self.v_target_traj_gpu[nt + 1])

                    with torch.no_grad():
                        noise = torch.randn_like(self.v_target_traj_gpu[nt + 1]) * 0.1
                    loss = self.loss_function(self.v, self.v_target_traj_gpu[nt + 1] + noise)

                    if (not torch.isfinite((loss)).all()):
                        print("dead by not finite number")
                        exit()
                    else:
                        traj_loss += loss

                with torch.no_grad():
                    if validity:
                        losses.append(traj_loss.item())
                if validity:
                    traj_loss.backward()

                self.grad_clip_models()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.x.detach_()
                self.v.detach_()

            print("# of valid Traj", len(losses))
            self.loss_each_epoch.append(sum(losses) / len(losses))

            if export_period != 0 and epoch % export_period == 0:
                self.export_traj(result_path[self.num_pos_sequence])
                self.make_checkpoint(EPOCH=epoch, LOSS=sum(losses) / len(losses),
                                     PATH=result_path[len(result_path) - 1])
                self._eval(EPOCH=epoch, result_path=result_path, max_iter=max_iter, export_period=export_period)

            self.export_loss_each_epoch(result_path[self.num_pos_sequence], max_iter, export_period)

            end_iter_epoch = time.time()
            print(
                f'Epoch {epoch + 1} / {max_iter}, total loss: {sum(losses) / len(losses)}, {end_iter_epoch - start_iter_epoch:.5f} sec / iter')

            with torch.no_grad():
                if (math.isnan(sum(losses))):
                    print("The loss is NaN.")

            # teacher_forcing_timer.step()
            # if epoch < teacher_forcing_timer.scheduler.period:
            #     teacher_forcing_timer.step()
            # else :
            #     teacher_forcing_timer.teacher_forcing_timer = -9999999 # after epoch
            self.scheduler.step()
            self.optimizer.zero_grad()

        self._eval(EPOCH=999999999, result_path=result_path, max_iter=max_iter, export_period=export_period)
        end_param = time.time()
        print(f"total time : {end_param - start_param:.5f} sec")

    def param_optimize_Procedural(self, result_path,min_i, export_period=10,iter_list_realoc_period = 100):

        iter_list_realoc_period = iter_list_realoc_period
        min_iter = min_i
        self.iter_list = [min_iter] * self.endFrame

        max_iter = self.opt_iter
        teacher_forcing = self.teacher_forcing


        self.DO_LEARNING = True
        start_param = time.time()
        print("====== OPTIMIZE PARAMETER======")

        self.hist_GT_velocity()

        for epoch in range(max_iter):
            self.set_models_trainMode()

            start_iter_epoch = time.time()
            losses = []

            window_size = teacher_forcing(epoch)

            start_frame_last_seq = self.endFrame - window_size
            start_frame_idx = list(range(start_frame_last_seq + 1))
            random.shuffle(start_frame_idx)

            print("winsize", window_size)

            for start_frame in start_frame_idx:

                traj_loss = 0.0
                validity = True

                if start_frame == 0:
                    self.init_forward()
                else:
                    self.x = self.x_target_traj_gpu[start_frame].clone()
                    self.v = self.v_target_traj_gpu[start_frame].clone()

                for nt in range(start_frame, start_frame + window_size):
                    self.step_learning_neural_constraint_procedural(nt)

                    loss = self.loss_function(self.v, self.v_target_traj_gpu[nt + 1])

                    if (not torch.isfinite((loss)).all()):
                        print("dead by not finite number")
                        exit()
                    else:
                        traj_loss += loss

                with torch.no_grad():
                    if validity:
                        losses.append(traj_loss.item())
                if validity:
                    traj_loss.backward()

                self.grad_clip_models()

                self.optimizer.step()
                self.optimizer.zero_grad()

                self.x.detach_()
                self.v.detach_()

            print("# of valid Traj", len(losses))
            self.loss_each_epoch.append(sum(losses) / len(losses))

            if export_period != 0 and epoch % export_period == 0:
                self.export_traj(result_path[self.num_pos_sequence])
                self.make_checkpoint(EPOCH=epoch, LOSS=sum(losses) / len(losses),
                                     PATH=result_path[len(result_path) - 1])
                self._eval(EPOCH=epoch, result_path=result_path, max_iter=max_iter, export_period=export_period)

            self.export_loss_each_epoch(result_path[self.num_pos_sequence], max_iter, export_period)

            end_iter_epoch = time.time()
            print(
                f'Epoch {epoch + 1} / {max_iter}, total loss: {sum(losses) / len(losses)}, {end_iter_epoch - start_iter_epoch:.5f} sec / iter')

            with torch.no_grad():
                if (math.isnan(sum(losses))):
                    print("The loss is NaN.")

            # teacher_forcing_timer.step()
            # if epoch < teacher_forcing_timer.scheduler.period:
            #     teacher_forcing_timer.step()
            # else :
            #     teacher_forcing_timer.teacher_forcing_timer = -9999999 # after epoch
            self.scheduler.step()
            self.optimizer.zero_grad()

            if(epoch%iter_list_realoc_period == 0 and epoch > 0):
                self.iter_list_realoc(self.num_solver_iter,min_iter)

        self._eval(EPOCH=999999999, result_path=result_path, max_iter=max_iter, export_period=export_period)
        end_param = time.time()
        print(f"total time : {end_param - start_param:.5f} sec")


    def iter_list_realoc(self,max_iter,min_iter):

        print("iter_list_realoc...")
        losses_gx = []

        self.iter_list = [min_iter] * self.endFrame

        with torch.no_grad():
            self.init_forward()
            for nt in tqdm(range(self.endFrame)):
                self.step_learning_neural_constraint(nt)
                gx = self.physics_informed_energy(self.v, self.v_pred)
                gx = torch.sqrt(gx)

                losses_gx.append(gx.item())


        num_levels = max_iter - min_iter + 1
        energy_min = min(losses_gx)
        energy_max = max(losses_gx)

        if energy_max == energy_min:
            print("iter_list_realoc::something goes wrong...")
            print(losses_gx)
            return [min_iter] * len(losses_gx)

        bin_size = (energy_max - energy_min) / num_levels

        iter_list = []
        for energy in losses_gx:
            bin_index = int((energy - energy_min) / bin_size)

            bin_index = min(bin_index, num_levels - 1)
            iter_value = min_iter + bin_index
            iter_list.append(iter_value)
        print("asdf")
        print(losses_gx)
        print(iter_list)

        self.iter_list = iter_list

    def param_optimize_TBPTT_stiffness(self, result_path, export_period=10):

        lr = 0.1
        opt_iter = 800
        max_iter = opt_iter
        lr_min = lr * 0.05

        self.k_opt_mul =torch.tensor([4], dtype=torch.float32, device='cuda',requires_grad=True)
        opt_param = [
            {'params': [self.k_opt_mul],
             'lr': lr},
        ]
        self.optimizer_stf = torch.optim.Adam(opt_param)
        self.scheduler_stf = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer_stf, T_max=opt_iter, eta_min=lr_min)

        teacher_forcing = lambda epoch: 8

        self.DO_LEARNING = True
        start_param = time.time()
        print("====== OPTIMIZE PARAMETER======")

        # plin_traj,Lang_traj = self.LOG_GT_momentum()


        for epoch in range(max_iter):
            start_iter_epoch = time.time()
            losses = []
            window_size = teacher_forcing(epoch)

            start_frame_last_seq = self.endFrame - window_size
            start_frame_idx = list(range(start_frame_last_seq + 1))
            random.shuffle(start_frame_idx)

            for start_frame in start_frame_idx:

                traj_loss = 0.0
                validity = True

                if start_frame == 0:
                    self.init_forward()
                else:
                    self.x = self.x_target_traj_gpu[start_frame].clone()
                    self.v = self.v_target_traj_gpu[start_frame].clone()

                for nt in range(start_frame, start_frame + window_size):
                    self.step_stiff(nt)
                    loss = self.loss_function(self.x, self.x_target_traj_gpu[nt + 1])           # at each frame, for each vertex, velocity diff norm sqr,

                    if (not torch.isfinite((loss)).all()):
                        with torch.no_grad():
                            print("The loss is NaN at frame", nt)
                            print("IGNORE this sub-trajectory")
                            print((math.isnan(sum(losses))), loss.item())
                            for ii in range(len(losses)):
                                print(ii, " : ", losses[ii])
                            validity = False
                            break
                    else:
                        traj_loss += loss


                with torch.no_grad():
                    if validity:
                        losses.append(traj_loss.item())
                if validity:
                    traj_loss.backward()

                torch.nn.utils.clip_grad_norm_(self.k_opt_mul, self.grad_clip)

                self.optimizer_stf.step()
                self.optimizer_stf.zero_grad()

                self.x.detach_()
                self.v.detach_()

            with torch.no_grad():
                print(self.k_opt_mul)
            print("# of valid Traj", len(losses))
            self.loss_each_epoch.append(sum(losses) / len(losses))

            if export_period != 0 and epoch % export_period == 0:
                self.export_traj(result_path[self.num_pos_sequence])
                self.make_checkpoint(EPOCH=epoch, LOSS=sum(losses) / len(losses),
                                     PATH=result_path[len(result_path) - 1])
                self._eval_stiff(EPOCH=epoch, result_path=result_path, max_iter=max_iter, export_period=export_period)

            self.export_loss_each_epoch(result_path[self.num_pos_sequence], max_iter, export_period)

            end_iter_epoch = time.time()
            print(
                f'Epoch {epoch + 1} / {max_iter}, total loss: {sum(losses) / len(losses)}, {end_iter_epoch - start_iter_epoch:.5f} sec / iter')

            with torch.no_grad():
                if (math.isnan(sum(losses))):
                    print("The loss is NaN.")

            self.scheduler_stf.step()
            self.optimizer_stf.zero_grad()

        self._eval_stiff(EPOCH=999999999, result_path=result_path, max_iter=max_iter, export_period=export_period)
        end_param = time.time()
        print(f"total time : {end_param - start_param:.5f} sec")



    def _eval(self,EPOCH,result_path,max_iter,export_period):
        print("evaluation...")
        start_iter_epoch = time.time()

        self.set_models_evalMode()

        losses_pos = []
        losses_F = []
        losses_J = []
        losses_Sig = []

        target_counter = 0
        self.DO_LEARNING = False
        with torch.no_grad():
            self.init_forward()
            # for nt in tqdm(range(self.endFrame)):
            for nt in range(self.endFrame):
                self.step_learning_neural_constraint(nt)

                self.x_traj_gpu[nt + 1] = self.x

                if nt + 1 in self.targetframe:
                    loss_pos = ((self.x - self.x_target_traj_gpu[self.targetframe[target_counter]]) ** 2).sum(dim=1).mean()
                    loss_F = self.deformation_gradient_error(self.x,self.x_target_traj_gpu[self.targetframe[target_counter]])
                    loss_J = self.J_error(self.x,self.x_target_traj_gpu[self.targetframe[target_counter]])
                    loss_Sig = self.Sig_error(self.x,self.x_target_traj_gpu[self.targetframe[target_counter]])

                    losses_pos.append(loss_pos.item())
                    losses_F.append(loss_F.item())
                    losses_J.append(loss_J.item())
                    losses_Sig.append(loss_Sig.item())

                    target_counter = target_counter + 1

            self.loss_eval['pos'].append(sum(losses_pos))
            self.loss_eval['F'].append(sum(losses_F))
            self.loss_eval['J'].append(sum(losses_J))
            self.loss_eval['Sig'].append(sum(losses_Sig))


        if EPOCH == 999999999 :
            self.loss_each_epoch.append(sum(losses_pos))
            self.export_traj(result_path[self.num_pos_sequence])
            self.make_checkpoint(EPOCH = 999999999,LOSS=sum(losses_pos),PATH=result_path[len(result_path)-1])
            # self.export_loss_each_epoch(result_path[len(result_path) - 1], max_iter, export_period)
            self.export_loss_eval(result_path[len(result_path) - 1], max_iter, export_period)
        else :
            self.export_loss_eval(result_path[len(result_path) - 1], max_iter, export_period)
            self.DO_LEARNING = True

        self.x.detach_()
        self.v.detach_()

        end_iter_epoch = time.time()
        print(f'Result:: position_loss: {sum(losses_pos)}, {end_iter_epoch - start_iter_epoch:.5f} sec / iter')


    def _eval_stiff(self,EPOCH,result_path,max_iter,export_period):
        print("evaluation...")

        self.DO_LEARNING = False

        start_iter_epoch = time.time()
        losses_pos = []

        target_counter = 0
        with torch.no_grad():
            self.init_forward()
            for nt in range(self.endFrame):
                self.step_stiff(nt)
                self.x_traj_gpu[nt + 1] = self.x

                if nt + 1 in self.targetframe:
                    loss_pos = ((self.x - self.x_target_traj_gpu[self.targetframe[target_counter]]) ** 2).sum(dim=1).mean()
                    losses_pos.append(loss_pos.item())
                    target_counter = target_counter + 1

            self.loss_eval['pos'].append(sum(losses_pos))

        self.x.detach_()
        self.v.detach_()

        end_iter_epoch = time.time()
        print(f'Result:: position_loss: {sum(losses_pos)}, {end_iter_epoch - start_iter_epoch:.5f} sec / iter')

    def get_R(self, F):

        U, sig, Vt = torch.linalg.svd(F)

        R_det = torch.det(U) * torch.det(Vt)
        refl_mask = R_det < 0.0
        Vt_adj = Vt.clone()
        Vt_adj[refl_mask, -1, :] *= -1

        return U @ Vt_adj

    def get_R_diff(self, F):

        Sig = torch.linalg.svdvals(F)
        tr = torch.sum(Sig, dim=1, keepdim=True)
        Ic = torch.sum(F ** 2, dim=(1, 2)).unsqueeze(-1)

        J = torch.det(F).unsqueeze(-1)

        FTF = F.transpose(-2, -1) @ F

        gI1 = 2 * F
        gI2 = 4 * (F @ (FTF))

        f1 = F[:, :, 0]
        f2 = F[:, :, 1]
        f3 = F[:, :, 2]
        P1 = torch.cross(f2, f3, dim=1)
        P2 = torch.cross(f3, f1, dim=1)
        P3 = torch.cross(f1, f2, dim=1)
        gJ = torch.stack((P1, P2, P3), dim=2)

        const = 4 * (tr ** 3) - (4 * Ic * tr) - (8 * J)

        with torch.no_grad():
            const_mask = torch.abs(const).squeeze(-1) < 1e-6

        const[const_mask, :] = 1e-6

        gtr1 = (2 * tr * tr + 2 * Ic) / const
        gtr2 = -2 / const
        gtrJ = (8 * tr) / const

        R = (gtr1.unsqueeze(-1) * gI1 + gtr2.unsqueeze(-1) * gI2 + gtrJ.unsqueeze(-1) * gJ)
        R[const_mask, :, :] = torch.eye(3, device='cuda')

        return R

    def position_error(self, pred, target):

        return ((pred - target) ** 2).sum(dim=1).mean() # this is strict mean squared error



    def deformation_gradient_error(self, pred, target):

        xli_pred = pred[self.t2v_i] - pred[self.t2v_l]
        xlj_pred = pred[self.t2v_j] - pred[self.t2v_l]
        xlk_pred = pred[self.t2v_k] - pred[self.t2v_l]
        Ds_pred = torch.stack((xli_pred, xlj_pred, xlk_pred), dim=2)

        xli_target = target[self.t2v_i] - target[self.t2v_l]
        xlj_target = target[self.t2v_j] - target[self.t2v_l]
        xlk_target = target[self.t2v_k] - target[self.t2v_l]
        Ds_target = torch.stack((xli_target, xlj_target, xlk_target), dim=2)


        F_diff = ((Ds_pred - Ds_target) @ self.invDMT) ** 2

        return F_diff.mean()


    def J_error(self,pred,target):

        xli_pred = pred[self.t2v_i] - pred[self.t2v_l]
        xlj_pred = pred[self.t2v_j] - pred[self.t2v_l]
        xlk_pred = pred[self.t2v_k] - pred[self.t2v_l]
        Ds_pred = torch.stack((xli_pred, xlj_pred, xlk_pred), dim=2)

        xli_target = target[self.t2v_i] - target[self.t2v_l]
        xlj_target = target[self.t2v_j] - target[self.t2v_l]
        xlk_target = target[self.t2v_k] - target[self.t2v_l]
        Ds_target = torch.stack((xli_target, xlj_target, xlk_target), dim=2)

        F_pred = Ds_pred @ self.invDMT
        F_target = Ds_target @ self.invDMT

        J_pred = torch.det(F_pred)
        J_target = torch.det(F_target)

        J_diff = (J_pred-J_target) ** 2

        return J_diff.mean()
    def Sig_error(self,pred,target):

        xli_pred = pred[self.t2v_i] - pred[self.t2v_l]
        xlj_pred = pred[self.t2v_j] - pred[self.t2v_l]
        xlk_pred = pred[self.t2v_k] - pred[self.t2v_l]
        Ds_pred = torch.stack((xli_pred, xlj_pred, xlk_pred), dim=2)

        xli_target = target[self.t2v_i] - target[self.t2v_l]
        xlj_target = target[self.t2v_j] - target[self.t2v_l]
        xlk_target = target[self.t2v_k] - target[self.t2v_l]
        Ds_target = torch.stack((xli_target, xlj_target, xlk_target), dim=2)

        F_pred = Ds_pred @ self.invDMT
        F_target = Ds_target @ self.invDMT

        Sig_pred = torch.linalg.svdvals(F_pred)
        Sig_target = torch.linalg.svdvals(F_target)

        trSig_pred = torch.sum(Sig_pred, dim=1, keepdim=True)
        trSig_target = torch.sum(Sig_target, dim=1, keepdim=True)

        trSig_diff = (trSig_pred-trSig_target) ** 2

        return trSig_diff.mean()


    def step_learning_neural_constraint_generalization_task_largedt_PBD(self,frame):

        # self.move_fixed_point(frame)
        self.f_ext = self.m_rep * self.g

        self.Lagrange = torch.zeros_like(self.Lagrange)

        self.y = self.x + self.dt_sub * (self.v + self.dt_sub * self.m_inv_rep * self.f_ext)

        #### Projection
        for st in range(self.num_solver_iter):
            xli = self.y[self.t2v_i] - self.y[self.t2v_l]
            xlj = self.y[self.t2v_j] - self.y[self.t2v_l]
            xlk = self.y[self.t2v_k] - self.y[self.t2v_l]

            Ds = torch.stack((xli, xlj, xlk), dim=2)
            F = Ds @ self.invDm

            C, dCdF = self.get_C_dCdx_LNC(F)
            gradC_x = dCdF @ self.invDMT

            grad_xi = gradC_x[:, :, 0]
            grad_xj = gradC_x[:, :, 1]
            grad_xk = gradC_x[:, :, 2]
            grad_xl = -(grad_xi + grad_xj + grad_xk)

            norm_sqr_grad_i = torch.sum(grad_xi ** 2, dim=1, keepdim=True)
            norm_sqr_grad_j = torch.sum(grad_xj ** 2, dim=1, keepdim=True)
            norm_sqr_grad_k = torch.sum(grad_xk ** 2, dim=1, keepdim=True)
            norm_sqr_grad_l = torch.sum(grad_xl ** 2, dim=1, keepdim=True)

            schur = self.m_inv_i * norm_sqr_grad_i + self.m_inv_j * norm_sqr_grad_j + self.m_inv_k * norm_sqr_grad_k + self.m_inv_l * norm_sqr_grad_l

            alpha = 1.0 / self.vol0.unsqueeze(-1)
            ld = - (self.dt_sub ** 2) * (C + alpha * self.Lagrange) / ((self.dt_sub ** 2) * schur + alpha)

            self.Lagrange = self.Lagrange + ld

            dx_i = self.m_inv_i * ld * grad_xi
            dx_j = self.m_inv_j * ld * grad_xj
            dx_k = self.m_inv_k * ld * grad_xk
            dx_l = self.m_inv_l * ld * grad_xl

            dx_ijkl = torch.cat((dx_i, dx_j, dx_k, dx_l), dim=0)
            self.dx = torch.zeros_like(self.dx, device='cuda').index_add(0, self.t2v_ijkl, dx_ijkl)

            self.y = self.y + self.dx / self.relaxation

            # hard coding floor collision
            floor_y = -5.0
            pos_y = self.y[:, 1]
            pos_mask = pos_y < floor_y

            if pos_mask.any():
                g = pos_y - floor_y  # [N]

                denom = (self.dt_sub ** 2) * self.m_inv[:, 0]
                ld_col = (self.dt_sub ** 2) * (-g) / denom  # [N]

                dx_col = torch.zeros_like(self.y)
                dx_col[:, 1] = self.m_inv[:, 0] * ld_col  # collision_y
                dx_col[~pos_mask] = 0
                self.y = self.y + dx_col / self.relaxation



        #### Update
        self.v = 0.999*(self.y - self.x) / self.dt_sub
        self.x = self.y


    def init_largedt_show(self):
        print("init_generalization!!!!!")
        with torch.no_grad():
            self.x = self.x0.clone()
            self.v = self.v0.clone()
            self.x_traj_gpu[0] = self.x.clone()

        epsilon = 1e-4
        col = self.x[:, 1]
        max_val = col.max()
        min_val = col.min()
        top_mask = (col - max_val).abs() <= epsilon
        bottom_mask = (col - min_val).abs() <= epsilon
        self.top_indices = torch.nonzero(top_mask, as_tuple=False).squeeze()
        self.bottom_indices = torch.nonzero(bottom_mask, as_tuple=False).squeeze()
        self.g = torch.tensor([0.0, -9.8, 0.0]).repeat(self.num_verts).reshape((-1, 3)).cuda()