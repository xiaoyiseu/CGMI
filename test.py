import torch, os, time
from module.TrainValid import evaluate, print_metrics
from utils.options import get_args
from model.CrossAttention import FeatureExtractor
from module.manager import Data_Indices
from utils.visualization import plot_pr_curve
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'

if __name__ == '__main__':
    args = get_args()
    args.output_dir = os.path.join(args.output_dir, 
                                   args.backbone+'_'
                                   + args.ImputMode+'_'
                                   +'ProbW'+str(args.ProbW)+ str(args.n_comp)+'_'
                                #    +'tail'+str(args.quantile)+'_'
                                   +'loss'+str(args.loss)+'_'
                                   +'prob'+str(args.prob)+'_'
                                #    +'filt'+str(args.filt)+'_'
                                   +'SA'+str(args.SA)+'_'
                                   +'SPNPairs'+str(args.SPNPairs)+'_'
                                   +'SFD'+str(args.SFD)+'_'
                                #    +'ProbW'+str(args.ProbW)+'_'
                                   +'CMF'+str(args.CMF)+'_'
                                   +'temp'+str(args.temp)
                                   )
    
    args.bestmodel = os.path.join(args.output_dir, args.bestmodel)
    args.vsmodel = os.path.join(args.output_dir, args.vsmodel)
    model0 = torch.load(args.bestmodel)
    modelVS0 = torch.load(args.vsmodel)


    args.mode = 'test'

    start_time = time.time()  
    test_loader, dic1, dic2 = Data_Indices(args)
    end_time = time.time() 

    meters_test, classification_metrics, RecogTine, true_s, probs_s, true_d, probs_d, cc_feat, vs_feat = evaluate(args, test_loader, model0, modelVS0)
    print_metrics(classification_metrics, 
                  file_path=os.path.join(args.output_dir,'test.txt'), mode = args.mode)

    N = len(true_s)
    EncTime = (end_time - start_time)/N
    print(f"测试集数量:  {N}")    
    print(f"Encode time: {EncTime:.4f} s, Recog time: {RecogTine:.4f} ms, total time: {EncTime*1000 + RecogTine:.4f} ms")


    plot_pr_curve(
        true_labels=true_s,
        pred_probs=probs_s,
        class_dict = dic1,
        task_name="Severity",
        file_path=os.path.join(args.output_dir, 'pr_curve_severity.svg')
    )
    plot_pr_curve(
        true_labels=true_d,
        pred_probs=probs_d,
        class_dict = dic2,
        task_name="Depart",
        file_path=os.path.join(args.output_dir, 'pr_curve_department.svg')
    )



DepartLib={'Internal Medicine':'IM', 
               'Obstetrics':'OB', 
               'Surgery':'SURG', 
               'Ophthalmology':'OPHTH', 
               'Gynecology':'GYN', 
               'Otolaryngology':'ORL',  
               'Neurosurgery':'NS', 
               'Trauma Center':'TC',
               'Orthopedics':'OPS',
               'Neurology':'NL'
              }   