class SequenceStreamer:
    def __init__(self, df, target_column, seq_length=35):
        self.df = df.dropna().reset_index(drop=True)
        self.target_column = target_column
        self.seq_length = seq_length
        self.feature_columns = [col for col in df.columns if col != target_column]
        self.df_features = self.df[self.feature_columns]
        self.current_index = seq_length - 1
        self.max_index = len(self.df) - 1

    def has_next(self):
        return self.current_index <= self.max_index

    def next_sequence(self):   # Generuojam nauja seka

        if self.has_next():
            start = self.current_index - (self.seq_length - 1)
            end = self.current_index + 1
            x_seq = self.df_features.iloc[start:end].copy().values
            self.current_index += 1
            return x_seq
        else:
            raise StopIteration("Nebėra daugiau sekų.")
        
    #modifikavimui pasirinkto stulpelio
    def modify_sequence(self, x_seq, feature_name, new_value):
        if feature_name not in self.feature_columns:
            raise ValueError(f"Stulpelis '{feature_name}' nerastas tarp bruožų.")
        feature_index = self.feature_columns.index(feature_name)
        x_seq[:, feature_index] = new_value
        return x_seq
    
    #pasiimam norimo stulpelio reiksme is eilutes - paskutines
    def get_feature_value_from_last_row(self, x_seq, feature_name):
        if feature_name not in self.feature_columns:
            raise ValueError(f"Stulpelis '{feature_name}' nerastas.")
        feature_index = self.feature_columns.index(feature_name)
        return x_seq[-1, feature_index]
    

#_________________________________________panaudojimas_________________________________
# # x = streamer.next_sequence()

# # # Pakeiti VISAS 'Temperature' reikšmes sekoje į 55.0
# # x = streamer.modify_sequence(x, feature_name='Temperature', new_value=55.0)
# x = streamer.next_sequence()
# reiksme = streamer.get_feature_value_from_last_row(x, "PV_torque")
# print("Reikšmė iš paskutinės eilutės:", reiksme)