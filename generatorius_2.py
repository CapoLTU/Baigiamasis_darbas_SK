class SequenceStreamer:
    def __init__(self, df, target_column, seq_length=35):                              
        self.df = df.dropna().reset_index(drop=True)                                   # Pasaliname NaN reiksmes ir atstatome eilučių indeksavimą
        self.target_column = target_column                                             # Prognozuojamu reismiu stulpelis
        self.seq_length = seq_length
        self.feature_columns = [col for col in df.columns if col != target_column]     # Paimami ivesties stulpeliai be Target
        self.df_features = self.df[self.feature_columns]                               # Issaugomi ivesties stulpeliai be Target
        self.current_index = seq_length - 1                                            # Sekos pradzios indeksas
        self.max_index = len(self.df) - 1                                              # Maksimalus galimas sekos indeksas

    def has_next(self):                                                                # Tikriname ar dar galime generuoti nauja seka
        return self.current_index <= self.max_index

    def next_sequence(self):                                                           # Generuojam nauja seka

        if self.has_next():
            start = self.current_index - (self.seq_length - 1)
            end = self.current_index + 1
            x_seq = self.df_features.iloc[start:end].copy().values                    # Paimami duomenys
            self.current_index += 1
            return x_seq
        else:
            raise StopIteration("Nebėra daugiau sekų.")                               # Isvedamas pranesimas ir stabdome cikla
        
    # Modifikuojame konkretaus požymio reikšmes visoje sekos eilutėje
    def modify_sequence(self, x_seq, feature_name, new_value):      
        if feature_name not in self.feature_columns:
            raise ValueError(f"Stulpelis '{feature_name}' nerastas tarp bruožų.")
        feature_index = self.feature_columns.index(feature_name)                      # Randame, kuriame stulpelyje yra modifikuotinas pozymis
        x_seq[:, feature_index] = new_value                                           # Pakeiciame visos sekos stulpelio reiksmes naujomis
        return x_seq
    
   
    def get_feature_value_from_last_row(self, x_seq, feature_name):                   # Paimama norimo stulpelio reiksme is paskutines eilutes 
        if feature_name not in self.feature_columns:
            raise ValueError(f"Stulpelis '{feature_name}' nerastas.")
        feature_index = self.feature_columns.index(feature_name)
        return x_seq[-1, feature_index]