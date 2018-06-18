import os

# Training dataset.............................
data = 'data'
train_data = "data/train"
test_data = "data/test"

# Training setting.............................
batch_size = 100
# student_follow: 
#       After the number of batches teacher trained, student will use teachers output to train
student_follow = 20 # batches
class token:
    def __init__(self):
        self.resource = student_follow
        self.teacher = student_follow
        self.student = -1

    def teacher_turn(self):
        if self.teacher == -1: return False
        else:
            self.teacher -= 1
            if self.teacher == 0:
                self.teacher = -1
                self.student = self.resource
            return True

    def student_turn(self):
        if self.student == -1: return False
        else:
            self.student -= 1
            if self.student == 0:
                self.student = -1
                self.teacher = self.resource
            return True
            
# Load model..................................
load_model = None


def dir_init():
    """To init dirs in setting.py """
    params = [data, train_data, test_data]
    for path in params:
        if not os.path.exists(path):
            os.mkdir(path)
    pass
