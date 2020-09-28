import numpy as np
import tensorflow as tf 
import pandas as pd 
from feature_selector import FeatureSelector
from sklearn.ensemble import RandomForestClassifier

class RandomForestAlgo(QCAlgorithm):
    
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2020, 9, 19)
        
        self.SetCash(10000000)
        tsla = self.AddEquity("TSLA", Resolution.Daily)
        spy = self.AddEquity("SPY", Resolution.Daily)
        
        self.symbols = [spy.Symbol,tsla.Symbol]
        self.lookback = 500
        
        
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Tuesday), self.TimeRules.AfterMarketOpen("TSLA",28), self.TrainForest)
        self.Schedule.On(self.DateRules.Every(DayOfWeek.Tuesday), self.TimeRules.AfterMarketOpen("TSLA",30), self.TradeStock)
        self.Schedule.On(self.DateRules.EveryDay("TSLA"), self.TimeRules.BeforeMarketClose("TSLA",10), self.TakeProfit)
        
    
    def exp_smooth_data(self,data):
            
        exp_smoothed_list = [] 
            
        exp_smoothed_list.append(data.iloc[0])
           
        for v in data[1:]:
            st = 0.2 * v + (1.0 - 0.2) * exp_smoothed_list[-1]
            exp_smoothed_list.append(st)  
            
        return exp_smoothed_list
            
        
    
    def comprised_smoothed_data(self,data_smoothed):
    
        tsla_data = data_smoothed.loc["TSLA"]
        
        sp_data = data_smoothed.loc["SPY"]
            
        df_smoothed = pd.DataFrame()
            
        close_smooth = self.exp_smooth_data(tsla_data['close'])
    
        df_smoothed['tsla_close_sm'] = close_smooth
    
        open_smooth = self.exp_smooth_data(tsla_data['open'])
        
        df_smoothed['tsla_open_sm'] = open_smooth
        
        high_smooth = self.exp_smooth_data(tsla_data['high'])
        
        df_smoothed['tsla_high_sm'] = high_smooth
        
        low_smooth = self.exp_smooth_data(tsla_data['low'])
        
        df_smoothed['tsla_low_sm'] = low_smooth
        
        volume_smooth = self.exp_smooth_data(tsla_data['volume'])
        
        df_smoothed['tsla_volume_sm'] = volume_smooth
        
        sp_close_smooth = self.exp_smooth_data(sp_data['close'])
            
        df_smoothed['sp_close_sm'] = sp_close_smooth
            
        return df_smoothed
            
    
    def computeRSI(self,data):
        
        diff = data.diff(1).dropna()        

        up_chg = 0 * diff
        down_chg = 0 * diff
        
        up_chg[diff > 0] = diff[ diff > 0]
        
        down_chg[diff < 0] = diff[ diff < 0 ]
    
        up_chg_avg   = up_chg.ewm(com=14-1 , min_periods=14).mean()
        down_chg_avg = down_chg.ewm(com=14-1 , min_periods=14).mean()
        
        rs = abs(up_chg_avg/down_chg_avg)
        rsi = 100 - 100/(1+rs)
        
        return rsi
    
    def calculate_features(self,smoothed_data):
        
        df_smoothed = smoothed_data
        
        df_smoothed['rsi'] = self.computeRSI(df_smoothed['tsla_close_sm'])
        df_smoothed['william'] = (df_smoothed['tsla_high_sm'].rolling(14).max() - df_smoothed['tsla_close_sm'])/(df_smoothed['tsla_high_sm'].rolling(14).max() - df_smoothed['tsla_low_sm'].rolling(14).min()) * -100
        df_smoothed['stch_osc'] = 100 * (df_smoothed['tsla_close_sm'] - df_smoothed['tsla_low_sm'].rolling(14).min())/(df_smoothed['tsla_high_sm'].rolling(14).max() - df_smoothed['tsla_low_sm'].rolling(14).min())
        df_smoothed['price_rate_change'] = (df_smoothed['tsla_close_sm'] - df_smoothed['tsla_close_sm'].shift(14))/df_smoothed['tsla_close_sm'].shift(14)
        df_smoothed['log_price'] = np.log(df_smoothed['tsla_close_sm'])
        df_smoothed['log_mov'] = df_smoothed['log_price'].rolling(6).mean()
        df_smoothed['log_diff'] = df_smoothed['log_price'] - df_smoothed['log_mov']
        df_smoothed['fast_mov'] = df_smoothed['tsla_close_sm'].rolling(3).mean()
        df_smoothed['slow_mov'] = df_smoothed['tsla_close_sm'].rolling(7).mean()
        df_smoothed['mov_diff'] = df_smoothed['fast_mov'] - df_smoothed['slow_mov']
        df_smoothed['mac_fast'] = df_smoothed['tsla_close_sm'].rolling(7).mean()
        df_smoothed['mac_slow'] = df_smoothed['tsla_close_sm'].rolling(20).mean()
        df_smoothed['mac_diff'] = df_smoothed['mac_fast'] - df_smoothed['mac_slow']
        df_smoothed['volume_log'] = np.log(df_smoothed['tsla_volume_sm'])
        df_smoothed['pct_change'] = df_smoothed['tsla_close_sm'].pct_change()
        df_smoothed['z_score'] = (df_smoothed['tsla_close_sm'] - df_smoothed['tsla_close_sm'].rolling(7).mean())/df_smoothed['tsla_close_sm'].std()
        df_smoothed['sp_return'] = df_smoothed['sp_close_sm'].pct_change(14)
        df_smoothed['return_two_week'] = df_smoothed['tsla_close_sm'].pct_change(14)
        df_smoothed['return_day'] = df_smoothed['tsla_close_sm'].pct_change(1)
        df_smoothed['return_month'] = df_smoothed['tsla_close_sm'].pct_change(5)
        df_smoothed['return_two_day'] = df_smoothed['tsla_close_sm'].pct_change(2)
        df_smoothed['return_week'] = df_smoothed['tsla_close_sm'].pct_change(5)
        df_smoothed['return_diff_sp'] = df_smoothed['return_day'] - df_smoothed['sp_return']
        df_smoothed['return_sp_std'] = df_smoothed['sp_return'].rolling(14).std()
        df_smoothed['return_std'] = df_smoothed['return_week'].rolling(14).std()
        df_smoothed['last_close'] = df_smoothed['tsla_close_sm'].shift(14)
        df_smoothed['last_open'] = df_smoothed['tsla_open_sm'].shift(14)
        df_smoothed['last_high'] = df_smoothed['tsla_high_sm'].shift(14)
        df_smoothed['last_low'] = df_smoothed['tsla_low_sm'].shift(14)
        df_smoothed['high_low'] = df_smoothed['last_high'] - df_smoothed['last_low']
        
        df_smoothed = df_smoothed.dropna()
        
        return df_smoothed
    
    def get_train_labels(self,smoothed_data):
        
        train_labels = smoothed_data
        
        train_labels['win'] = np.where((train_labels['tsla_close_sm'].shift(-5) > train_labels['tsla_close_sm']), 1, 0)
        
        train_labels = train_labels['win']
        
        return train_labels
    
    def get_train_features(self,smoothed_data):
        
        train_features = smoothed_data
        
        clean_train_features = train_features.drop(columns=['last_open', 'last_close', 'tsla_high_sm', 'tsla_low_sm','sp_close_sm', 
                                                            'volume_log', 'log_mov', 'slow_mov', 'mac_fast', 'mac_slow', 
                                                            'z_score', 'return_day', 'stch_osc', 'return_two_week', 'last_low', 
                                                            'return_week', 'last_high', 'tsla_open_sm', 'log_price', 'fast_mov']) 
                                                            
        return clean_train_features
        

    def feature_selection(self,train_features,train_labels):
        
        fs = FeatureSelector(data=train_features, labels=train_labels)
        
        fs.identify_collinear(correlation_threshold=0.975)
        
        fs.identify_zero_importance(task='regression',eval_metric ='auc',n_iterations=10,early_stopping=True)
        
        fs.identify_low_importance(cumulative_importance = 0.99)
        
        all_to_remove = fs.check_removal()
        
        clean_features = train_features.drop(columns = all_to_remove)
        
        return clean_features
        
    
    
    def TrainForest(self):
        
        symbols = ["TSLA","SPY"]
        
        history = self.History(symbols, self.lookback, Resolution.Daily)
        
        self.X_train, self.X_predict = [], []
        
        self.y_train = []
        
        self.buy_or_sell = []
            
        if not history.empty:
                
            get_smooth_data = self.comprised_smoothed_data(history)
                
            get_calc_features = self.calculate_features(get_smooth_data)
        
            train_test = self.get_train_features(get_calc_features)
        
            train_label = self.get_train_labels(get_calc_features)
                
            self.X_train = train_test
            
            last = len(train_test)-1
                
            self.X_predict = train_test.iloc[last:]
                
            self.y_train = train_label
            
 
        if not self.X_train.empty:
            classifier = RandomForestClassifier(n_estimators=3,max_depth=30,max_features='sqrt',min_samples_leaf=62,min_samples_split=2,random_state=42)
            classifier.fit(self.X_train,self.y_train)
            y_pred = classifier.predict(self.X_predict)
            self.buy_or_sell.append(y_pred)
            self.Debug(str(self.Time) + "Buy_or_sell:" + str(y_pred))
            
        
    def TradeStock(self):
        
        if self.Portfolio["TSLA"].IsShort:
            if self.buy_or_sell[-1] == 1:
                self.Liquidate()
                self.SetHoldings("TSLA",1)
        elif self.Portfolio["TSLA"].Invested:
            if self.buy_or_sell[-1] == 0:
                self.Liquidate()
                self.SetHoldings("TSLA",-1)
        elif not self.Portfolio["TSLA"].Invested:
            if self.buy_or_sell[-1] == 1:
                self.SetHoldings("TSLA",1)
            else:
                self.SetHoldings("TSLA",-1)
    
    def TakeProfit(self):
        
        if self.Portfolio["TSLA"].UnrealizedProfitPercent <= -0.02:
            self.Liquidate()
            self.Debug("Risk Parameter Met")
        if self.Portfolio["TSLA"].UnrealizedProfitPercent >= 0.75:
            self.Liquidate()
            self.Debug("Return Parameter Met")
