import argparse
import warnings

import torch
import pandas as pd
import numpy as np
from datetime import datetime,timedelta

from core.module import Seq2Seq,load_pickle


DEVICE = torch.device("cpu")


OXT_TO_COLOR = {'ROOM AIR' : 'green',
                'NASAL' : 'yellow',
                'MASK' : 'lightcoral',
                'HFNC' : 'red',
                'VENTILATION' : 'purple',
                'SYMPTOM' : 'purple'}

class covsf():
    def  __init__(self) -> None:
        #Required features
        self.all_features = ['BUN','Creatinine', 'Hemoglobin', 'LDH','Neutrophils','Lymphocytes','Platelet count',
            'Potassium', 'Sodium', 'WBC Count', 'CRP', 'BDTEMP', 'BREATH', 'DBP',
            'PULSE', 'SBP', 'SPO2','Oxygen']
        self.features = ['BUN', 'Creatinine', 'Hemoglobin', 'LDH', 'NLR', 'Platelet count',
            'Potassium', 'Sodium', 'WBC Count', 'CRP', 'BDTEMP', 'BREATH', 'DBP',
            'PULSE', 'SBP', 'SPO2','Oxygen']
        self.input_features = ['BUN', 'Creatinine', 'Hemoglobin', 'LDH', 'NLR', 'Platelet count',
            'Potassium', 'Sodium', 'WBC Count', 'CRP', 'BDTEMP', 'BREATH', 'DBP',
            'PULSE', 'SBP', 'SPO2']
        self.oxt = ['ROOM AIR','NASAL','MASK','HFNC','VENTILATION']

        # load model , scaler
        model_args = load_pickle('/mnt/models/code/core/module/model_arg.pkl') 
        model_args['test'] = True
        print(model_args)
        model = Seq2Seq(**model_args).to(DEVICE)
        model.load_state_dict(torch.load('/mnt/models/code/core/module/model.pt', map_location=DEVICE))
        model.eval()

        self.model = model
        self.scaler = load_pickle('/mnt/models/code/core/module/scaler.pkl')

    '''
    preprocess : Preprocessing dataframe
    1. Linear interpolation for laboratory data
    2. Calculate NLR 
    input df : pd.DataFrame
        type : pd.DataFrame
        shape : 1 x T x 18
            T(length of input days): min 1, max 5
            18 : Vital signs data (6) + Laboratory data(11) + Oxygen treatment(1)
                - Refer to the names and order of self.all_features
                - Oxygen treatment is not mandatory (only used for visualization)
                    values must be one of the self.oxt('ROOM AIR','NASAL','MASK','HFNC','VENTILATION')
        values : vital signs & laboratory data, must be recorded for all slots

    return : df : pd.DataFrame
    '''
    def preprocess(self,df: pd.DataFrame) -> pd.DataFrame:
        
        df = df.infer_objects()
        df = df.interpolate().ffill().bfill()
        df['NLR']= df['Neutrophils'] / df['Lymphocytes']
        df.drop(['Neutrophils','Lymphocytes'],axis=1,inplace=True) 
        df = df[self.features] # Reorder
        return df
    
    '''
    standardization : Standardization for input window X
    input X : torch.Tensor
        type : pd.DataFrame
        shape : 1 x T x 17
            T(length of input days): min 1, max 5
            17 : input_features (use self.input_features)
        values : vital signs & laboratory data, must be recorded for all slots

    return : torch.Tensor
    '''
    def standardization(self,x: torch.Tensor):
        mini_batch,t,fea_dim = x.shape

        x = self.scaler.transform(x.reshape(mini_batch * t,fea_dim).numpy())
        x = torch.from_numpy(x)
        x = x.reshape(mini_batch,t,fea_dim)
        return x

    '''
    _run : Prediction for only one input window
    input X : input window 
        type : pd.DataFrame
        shape : T x 16
            T(length of input days): min 1, max 5
            17 : input_features (use self.input_features)
        values : vital signs & laboratory data, must be recorded for all slots

    return : CovSF outputs for +0, +1, +2, +3
    '''
    def _run(self,x: pd.DataFrame) ->list: # Prediction for only one data
        with torch.no_grad():
            x = torch.Tensor(x.to_numpy()).unsqueeze(0)
            x = self.standardization(x).to(DEVICE)

            y0 = torch.Tensor([0.0,0.0]).reshape(1,1,2).to(DEVICE)

            pred = self.model(x,y0)
            pred = pred.squeeze(0).cpu().tolist()
            pred = [_[1] for _ in pred]

        return pred

    '''
    run : Prediction for whole hospitalization days
    input df : pd.DataFrame
        type : pd.DataFrame
        shape : 1 x T x 18
            T(length of input days): min 1, max 5
            19 : Vital signs data (6) + Laboratory data(11) + Oxygen treatment(1)
                - Refer to the names and order of self.all_features
                - Oxygen treatment is not mandatory (only used for visualization)
                    values must be one of the self.oxt('ROOM AIR','NASAL','MASK','HFNC','VENTILATION')
        values : vital signs & laboratory data, must be recorded for all slots

        save : if not False, save results at feded values. 
    return : df : pd.DataFrame
    '''
    def run(self,df: pd.DataFrame,save: str=False) ->list: 
        df = self.preprocess(df=df)

        days = df.index.tolist()
        targets = df['Oxygen'].tolist()
        last_day = datetime.strptime(days[-1].strftime('%Y-%m-%d'), '%Y-%m-%d')
        add_days = [(last_day+timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,4)]
        odays = days +add_days
        day_predict = {day : [] for day in odays}
        preds = []

        for i,day in enumerate(days):
            
            input_days = days[max(i-4,0):i+1]
            output_days = odays[i:i+4]
            input_df = df.loc[input_days,self.input_features]

            pred = self._run(input_df)
            preds.append(pred)

            for j,_day in enumerate(output_days):
                day_predict[_day].append(pred[j])

        # values = list(range(len(day_predict)))
        # res['CovSF'] = values[:len(day_predict)]


        res = pd.DataFrame(preds,index=days,columns=['+0','+1','+2','+3'])
        
        index_length = len(res.index)
        print(f"Index length: {index_length}")
        for day,ps in day_predict.items():
            day_predict[day] = np.array(ps).mean()

        #res = res.reindex(day_predict)


        res.index = pd.to_datetime(res.index)
        # res의 인덱스를 day_predict의 인덱스와 일치시킴
        res = res.reindex(day_predict.keys())
        res['CovSF'] = pd.Series(day_predict)
        res = res[['CovSF']]
        
        for i,add_day in enumerate(add_days):
            if add_day not in res.index:  # 이미 존재하는 날짜는 건너뛰기
                ps = day_predict.get(add_day, np.nan)  # add_day가 day_predict에 없으면 NaN으로 설정
                #ps = day_predict[add_day]
                res.loc[f"+{i+1}"] = ps 

        if save: res.to_csv(save)
        print("========res=========")
        print(res)
        return res

    

def main():
    parser = argparse.ArgumentParser(description='CovSF')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-r', action='store_true',help='Run CovSF')

    #INPUT PATH : required
    parser.add_argument('INPUT_FILE', type=str,help='Input file path')
    
    #SAVE PATH : optional
    parser.add_argument('SAVE_PATH',type=str,nargs='?',default=None,help='Save path. If None, results will be saved at current path')

    args = parser.parse_args()
    save_path = args.SAVE_PATH if args.SAVE_PATH else './'


    CovSF = covsf()

    save_path += 'results.csv'
    df = read_df(args.INPUT_FILE)
    CovSF.run(df,save_path)
   

def read_df(path):
    if path.lower().endswith('.csv'):
        return pd.read_csv(path, index_col=0)
    elif path.lower().endswith('.xlsx') or path.lower().endswith('.xls'):
        return pd.read_excel(path, index_col=0)
    else:
        raise ValueError("Unsupported file format. Only CSV and Excel files are supported.")

if __name__ == '__main__':
    
    main()