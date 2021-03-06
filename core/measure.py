# 3rd module
import tensorflow as tf
from pympler import asizeof
import matplotlib.pyplot as plt

def measure(data = None, name = 'NAN'):
    '''Format print an bytes size of input object'''
    with tf.device('cpu:1'):
        if data is None:
            raise Exception('None data can not be measure')
        print('Measure size: %s'.ljust(60, '.') %  name)
        size = asizeof.asizeof(data)
        print('\t%s bytes, \t%s KB, \t%s MB' % (size, size / 1024, size / 1048576))
        print()

def find_prediction(predict = None):
    '''
    Find prediction from softmax result
    return example: [0 , 0, 1, 0, 0]
    '''
    predict = list(predict)
    if predict is None: raise Exception('Prediction must given')

    tmp = [0 for _ in range(len(predict))]
    # set the highest probiblity to 1
    tmp[predict.index(max(predict))] = 1
    return tmp

def calculate_accuracy(predictions = None, labels = None):
    '''
    retrun accuracy of a list of prediction
    labels is a list of number
    '''
    if predictions is None or labels is None: raise Exception('Params not given enough')

    predictions = [find_prediction(predict) for predict in predictions]
    predict_correct = 0
    if type(labels[0]) == type([]):
        check_predict = lambda x, y: 1 if x.index(max(x)) == y.index(max(y)) else 0
    else:
        check_predict = lambda x, y: 1 if x.index(max(x)) == y else 0
    for idx, label in enumerate(labels):
        predict_correct += check_predict(predictions[idx], labels[idx])

    return (predict_correct / len(labels)) * 100

    
def calculate_prediction_match_rate(student_pres = None, teacher_pres = None):
    if student_pres is None or teacher_pres is None: raise Exception('Params not enough')

    student_pres = [find_prediction(pre) for pre in student_pres]
    teacher_pres = [find_prediction(pre) for pre in teacher_pres]

    check_predict = lambda x, y: 1 if x.index(max(x)) == y.index(max(y)) else 0
    prediction_correct = 0
    for idx, student_pre in enumerate(student_pres):
        prediction_correct += check_predict(student_pres[idx], teacher_pres[idx])

    return (prediction_correct / len(student_pres)) * 100

def format_plot_v2(
        datas = None, 
        save_name = None, 
        legends = [],
        title = 'Training accuarcy',
        xlabel = 'Epochs',
        ylabel = 'Accuracy'
        ):
    def deafult_arr(arr, idx):
        try:
            return arr[idx]
        except Exception as err:
            print(err)
            return None
    if datas is None: raise Exception('Datas must given')
    if len(datas) > 4: raise Exception('4 data is the max')
    if save_name is None: raise Exception('Please give save name')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    colors = ['r', 'b', 'g', 'c']
    styles = ['o', 's', '^', '*']

    for idx, data in enumerate(datas):
        plt.plot(data, label=deafult_arr(legends, idx) ,color = colors[idx], marker = styles[idx])
        
    plt.legend(legends, loc = 'best')
    plt.legend(loc = 'best')
    plt.savefig('images/%s'  % save_name)
    plt.close()

def format_plot(
    datas = None,
    legends = [],
    save_name = None, 
    title = 'Training accuarcy',
    xlabel = 'Epochs',
    ylabel = 'Accuracy'
    ):
    if datas is None or save_name is None: raise Exception('Parmas is not enough')

    for data in datas:
        plt.plot(data)
    plt.legend(legends, loc = 'best')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig('images/%s'  % save_name)
    plt.close()

def concat_history(his_a = None, his_b = None):
    if his_a is None or his_b is None: raise Exception('Parmas is not enough')

    keys = ['val_loss', 'val_acc', 'loss', 'acc']
    for key in keys:
        his_a.history[key] += his_b.history[key]

    return his_a