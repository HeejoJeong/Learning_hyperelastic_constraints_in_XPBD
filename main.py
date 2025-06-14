import torch
import taichi as ti

from Scene.scenes import scene_scaling as scene
from Scene.scenes import generalize as scene


from Util.mesh_exporter import export_obj_sequence_with_normals
import Util.keyboardWaiting

mode_dict = {"just_forward": 0, "forward_vanilla": 1, "param_opt": 2, "mod_stiff_opt": 3,
             "forward_models": 4, "compare_xpbd_pd": 5, "model_analysis": 6, "model_analysis_extern": 7, "param_further_opt": 8, "export_learned": 42,"export_xpbd": 43,"export_GT": 44
             ,"do_generalization_task":66}

mode_value = mode_dict.get(scene.mode)

FLAG_WIRE_FRAME = False
keyframe_dy = 0

sim = scene.sim


if mode_value == 0:
    sim.USE_NEURAL_CONSTRAINT = False
    sim.do_sim(scene.keyframePath, gen_keyframe=True)

elif mode_value == 1:
    sim.USE_NEURAL_CONSTRAINT = True
    sim.do_sim(scene.keyframePath, gen_keyframe=False)

elif mode_value == 2:
    sim.USE_NEURAL_CONSTRAINT = True
    sim.load_target(scene.keyframePath)
    Util.keyboardWaiting.waiting()
    sim.param_optimize_Procedural(result_path=scene.result_path,min_i = scene.min_solver_iter, export_period=scene.export_period,iter_list_realoc_period=100)
    # sim.param_optimize_TBPTT(result_path=scene.result_path, export_period=scene.export_period)

    exit()

elif mode_value == 4:
    sim.USE_NEURAL_CONSTRAINT = True
    sim.load_target(scene.keyframePath)
    sim.load_checkpoint(scene.load_path)
    sim.do_sim_evaluation(scene.keyframePath, gen_keyframe=False)


elif mode_value == 3:
    sim.load_target(scene.keyframePath)
    Util.keyboardWaiting.waiting()
    sim.param_optimize_TBPTT_stiffness(result_path=scene.result_path, export_period=scene.export_period)


elif mode_value == 5:
    sim.load_target(scene.keyframePath)
    sim.do_sim(scene.keyframePath, gen_keyframe=False)


elif mode_value == 42:
    sim.USE_NEURAL_CONSTRAINT = True
    sim.load_target(scene.keyframePath)
    sim.load_checkpoint(scene.load_path)
    sim.do_sim_evaluation(scene.keyframePath, gen_keyframe=False)

    faces_np = sim.mesh_dy.face_indices.to_numpy()
    faces_np = faces_np.reshape(faces_np.size // 3, 3)
    export_obj_sequence_with_normals(sim.x_traj_gpu,faces_np,"Learned")

elif mode_value == 43:
    sim.load_target(scene.keyframePath)
    sim.do_sim(scene.keyframePath, gen_keyframe=False)

    faces_np = sim.mesh_dy.face_indices.to_numpy()
    faces_np = faces_np.reshape(faces_np.size // 3, 3)
    export_obj_sequence_with_normals(sim.x_traj_gpu,faces_np,"XPBD")

elif mode_value == 44:
    sim.load_target(scene.keyframePath)
    faces_np = sim.mesh_dy.face_indices.to_numpy()
    faces_np = faces_np.reshape(faces_np.size // 3, 3)
    export_obj_sequence_with_normals(sim.x_target_traj_gpu,faces_np,"GT")


elif mode_value == 66:
    sim.USE_NEURAL_CONSTRAINT = True
    sim.load_target(scene.keyframePath)
    sim.load_checkpoint(scene.load_path)
    sim.do_sim_generalization_task(scene.keyframePath, gen_keyframe=False)

    faces_np = sim.mesh_dy.face_indices.to_numpy()
    faces_np = faces_np.reshape(faces_np.size // 3, 3)
    export_obj_sequence_with_normals(sim.x_traj_gpu,faces_np,"generalize")
else:
    print(scene.mode, ":", mode_value)
    print("Do nothing")
    exit()

Util.keyboardWaiting.waiting()


wx, wy = 1024, 768

window = ti.ui.Window("Diff", (wx, wy), fps_limit=60)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()

camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)


print("=====================================================")

frame = 0
run_flag = True

offset_flag = True

while window.running:

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):
        if window.event.key == ti.ui.UP:
            sim.load_next_pos_seq()
        if window.event.key == ti.ui.DOWN:
            sim.load_next_pos_seq(False)
        if window.event.key == ti.ui.SPACE:
            run_flag = not run_flag
        if window.event.key == 'y':
            FLAG_WIRE_FRAME = not FLAG_WIRE_FRAME
        if window.event.key == 'r':
            frame = 0
        if window.event.key == 'o':
            offset_flag = not offset_flag
        if window.event.key == 'k':
            keyframe_dy = (keyframe_dy + 1) % 3
        if window.event.key == ti.ui.RIGHT:
            frame = (frame + 1)% sim.endFrame
            print("current frame : ", frame)
        if window.event.key == ti.ui.LEFT:
            frame = (frame - 1) % sim.endFrame
            print("current frame : ", frame)

    sim.set_render_pos(frame, keyframe_dy,offset_flag)

    if keyframe_dy != 2:
        scene.mesh(sim.pos, indices=sim.mesh_dy.face_indices, color=(0.0, 0.0, 0.0), show_wireframe=True)
        scene.mesh(sim.pos, indices=sim.mesh_dy.face_indices, color=(0, 1.0, 0.0), show_wireframe=FLAG_WIRE_FRAME)

    if keyframe_dy:
        scene.mesh(sim.pos_target, indices=sim.mesh_dy.face_indices, color=(0, 0.0, 0.0), show_wireframe=True)
        scene.mesh(sim.pos_target, indices=sim.mesh_dy.face_indices, color=(1, 0.0, 0.0),
                   show_wireframe=FLAG_WIRE_FRAME)

    camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()

    if run_flag:
        frame = frame + 1

    # if frame % (sim.endFrame) == 0 :
    #     run_flag = False
    if frame % (sim.endFrame + 1) == 0:
        frame = 0
        sim.frame_ti = 0






