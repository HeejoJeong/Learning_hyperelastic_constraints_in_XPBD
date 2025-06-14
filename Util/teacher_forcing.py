
import math

class TeacherForcingTimer():
    class cosine_annealing_scheduler:
        def __init__(self, cos_annealing_param):

            period, w_min, w_max = cos_annealing_param
            print(f"TEACHERFORCING::COSINE_ANNEALING_SCHEDULER period:{period}, w_max:{w_min}, c_min{w_max}")

            self.period = period
            self.center = (w_max + w_min) * 0.5
            self.amp = (w_min-w_max) * 0.5
            self.current = 0

        def step(self):
            self.current = (self.current + 1) % self.period

        def get_frequency(self):
            return 1 / (self.center + self.amp * math.cos(math.pi*self.current / self.period))

    def __init__(self, cos_annealing_param):

        self.use_teacher_forcing = False
        if not cos_annealing_param:
            print(f"TEACHERFORCING::DO NOT USE TEACHER FORCING")
            return

        assert len(cos_annealing_param) == 3, "COSINE_ANNEALING_SCHEDULER PARAMETER ERROR"

        self.use_teacher_forcing = True

        self.scheduler = self.cosine_annealing_scheduler(cos_annealing_param)
        self.teacher_forcing_timer = self.scheduler.get_frequency()

    def check_forcing_timer(self):
        if not self.use_teacher_forcing :
            return False


        if self.teacher_forcing_timer > 1.0 - 1e-6 :
            self.teacher_forcing_timer = self.teacher_forcing_timer - 1.0
            self.teacher_forcing_timer = self.teacher_forcing_timer + self.scheduler.get_frequency()
            return True
        else :
            self.teacher_forcing_timer = self.teacher_forcing_timer + self.scheduler.get_frequency()
            return False



    def step(self):
        if not self.use_teacher_forcing :
            return

        self.teacher_forcing_timer = self.scheduler.get_frequency()
        self.scheduler.step()