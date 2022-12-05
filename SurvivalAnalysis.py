from lifelines import KaplanMeierFitter, NelsonAalenFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pandas as pd
import numpy as np
# from scipy.stats import chi2

import matplotlib.pyplot as plt

class SurvivalAnalysis:
    """
    Object that performs Survival Analysis
    
    Attributes
    ----------
    data_encoded : pd.DataFrame
        data only with numeric or categorical (up to 10 categories) columns.
    cat_code_dfs_dict : dict
        codes of the encoded categorical variables.
    signif_cox_coef : dict
        coefficients that were rendered significant after cox model fit.
    event : str
        name of the event column 
    time : str
        name of the time column 
    group_by : str
        name of the group_by column 
        
    Methods
    -------
    plot_survival_function(self, event = None, time = None, group_by = None)
        
    plot_cumulative_density(self, event = None, time = None, group_by = None)
    
    plot_hazard(self, event = None, time = None, group_by = None)
    
    cox_fitter(self, time = None, event = None, group_by = None, significance_level = 1)
    
    problemsolver(self)
    
    predict(self, time, model = "kmf", raw = False)
    
    test_hypothesis(self, time = None, event = None, group_by = None)
    """
        
    def __init__(self, data_file_path, event = None, time = None, group_by = None):
        """
        Initialized the object.
        
        Parameters
        ----------
        data_file_path : str
            Path to the CSV file
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        """
        
        self.data_file_path = data_file_path
        
        try: 
            self.data = pd.read_csv(data_file_path)
            self.data_encoded = pd.read_csv(data_file_path)
        except FileNotFoundError:
            print("Could not find specified file. Please, check it is in the directory.")
            return
        
        self.data.columns = [column.lower() for column in self.data.columns]
        self.data_encoded, self.cat_code_dfs_dict = self.__transform_categorical(self.data_encoded)
        self.signif_cox_coef = dict()
        
        if event and event not in self.data.columns:
            print("event is not in data columns")
        
        if time and time not in self.data.columns:
            print("time is not in data columns")
            
        if group_by and group_by not in self.data.columns:
            print("group_by is not in data columns")
            
        self.event = event
        self.time = time
        self.group_by = group_by
        self._should_terminate = False
        
    def plot_survival_function(self, event = None, time = None, group_by = None): 
        """
        Plots survival function. This function should know about event, time and/or group_by 
        columns. You should either provide them in the function call; otherwise, they should 
        be retrived from problemsolver() function call. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set() == False:
            return
        
        print("Kaplan-Meier Estimate")
            
        if self.group_by:
            for name, grouped_df in self.data.groupby(self.group_by):

                cols = self.__get_time_event(grouped_df)

                if cols:
                    time_variable = cols[0]
                    event_variable = cols[1]
                    
                    kmf = self.__fit_kmf(time_variable, event_variable, name)
                    
                    if kmf:
                        kmf.plot_survival_function()
        else:
            kmf = self.__fit_kmf(self.data_encoded[self.time], self.data_encoded[self.event])
            if kmf:
                kmf.plot_survival_function()
        
        plt.show()
           
    
    def plot_cumulative_density(self, event = None, time = None, group_by = None):
        """
        Plots cumulative density. This function should know about event, time and/or group_by 
        columns. You should either provide them in the function call; otherwise, they should 
        be retrived from problemsolver() function call. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set() == False:
            return
        
        print("\nCumulitive density")
        print("It gives us a probability of a person dying at a certain timeline.")
        
        if self.group_by:
            for name, grouped_df in self.data.groupby(self.group_by):
            
                cols = self.__get_time_event(grouped_df)

                if cols:

                    time_variable = cols[0]
                    event_variable = cols[1]

                    kmf = self.__fit_kmf(time_variable, event_variable, name)
                    
                    if kmf:
                        kmf.plot_cumulative_density()
                    else:
                        return
        else:
            
            kmf = self.__fit_kmf(self.data_encoded[self.time], self.data_encoded[self.event])
                    
            if kmf:
                kmf.plot_cumulative_density()
            
        plt.show()
        
        self.__print_divider()
    
    
    def plot_hazard(self, event = None, time = None, group_by = None):
        """
        Plots hazard function. This function should know about event, time and/or group_by 
        columns. You should either provide them in the function call; otherwise, they should 
        be retrived from problemsolver() function call. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set() == False:
            return
        
        print("\nHazard function")
        
        if self.group_by:
            
            for name, grouped_df in self.data.groupby(self.group_by):

                cols = self.__get_time_event(grouped_df)

                if cols:
                    time_variable = cols[0]
                    event_variable = cols[1]
                    
                    naf = self.__fit_naf(time_variable, event_variable, label = name)
                    
                    if naf:
                        try:
                            naf.plot_cumulative_hazard()
                        except:
                            pass
                    else:
                        return
        else:
            naf = self.__fit_naf(self.data_encoded[self.time], self.data_encoded[self.event])
            if naf:
                naf.plot_cumulative_hazard()
            else:
                return
                    
        plt.show()
        
        self.__print_divider()
    
    def cox_fitter(self, time = None, event = None, group_by = None, significance_level = 1):
        """
        This function runs Cox Fitter Analysis. This function should know about event, time 
        and/or group_by columns. You should either provide them in the function call; otherwise, 
        they should be retrived from problemsolver() function call. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        significance_level : float
            level of significance for the fitted coefficients
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set() == False:
            return
        
        print("\nCox Proportional Hazard Model Regression \n")
        
        cph = CoxPHFitter()
        
        try:
            cph.fit(self.data_encoded, self.time, event_col = self.event)
        except KeyError:
            print("ERROR: Could not find column")
            return;
        except TypeError:
            print("ERROR: Please, check your columns. Fit function cannot be executed.")
        
        significant_columns = cph.summary[cph.summary["p"] < significance_level]["exp(coef)"]
        
        coefs = {significant_columns.index[index] : column for index, column in enumerate(significant_columns)}
        coefs = {k : f"{int(100 * (v - 1))}%" for k, v in coefs.items()}
        
        self.signif_cox_coef = coefs
        
        for signif_col, percent in self.signif_cox_coef.items():
            if self.group_by == None or signif_col == self.group_by:
                try:
                    pivot_category = self.cat_code_dfs_dict[signif_col][0]
                    print(f"{pivot_category} has {percent} more risk to die.")
                except:
                    pass
        
        if self.group_by == None:
            cph.plot()
        
        self.__print_divider()
        
    
    def problemsolver(self):
        """
        Provides detailed Survival Analysis. 
        
        This function requests user to input event, time and group columns in the data.
        
        The function does not return anything, but prints the summary of the analysis.
        
        The function has calls to 
            self.plot_survival_function(_)
            self.plot_cumulative_density(_)
            self.plot_hazard(_)
            self.cox_fitter(_)
            
        Each of these functions can be called individually, this method simply gathers everything 
        together with friendly CLI.
        """
        
        if self.event == None or self.time == None or self.group_by == None:
            chosen_cols = []

            for needed_col in ["event", "time", "group by"]:

                columns = self.data.columns
                columns = list(set(columns) - set(chosen_cols))

                print(f"Please, choose {needed_col} column, (type in either column name or index)\n")

                for i in range(len(columns)):
                    print("\t", i, "|", columns[i])

                print()

                column_found = False

                while True:
                    try:
                        chosen_col = input()

                        if chosen_col in columns:
                            break
                        else:
                            index = int(chosen_col)
                            if index < len(columns):
                                chosen_col = columns[index]
                                break
                            else:
                                print(f"\nIndex {index} out of bounds\n")

                    except KeyboardInterrupt:
                        print("\nWill wait for you!")
                        return

                    except ValueError:
                        print(f"\nCould not find column {chosen_col}\n")

                print(f"\nChosen \"{chosen_col}\" as {needed_col} column")

                self.__print_divider()

                chosen_cols.append(chosen_col)

            self.event = chosen_cols[0]
            self.time = chosen_cols[1]
            self.group_by = chosen_cols[2]
        
        functions_to_run = [self.plot_survival_function,
                            self.plot_cumulative_density,
                            self.plot_hazard, 
                            self.cox_fitter,
                            self.test_hypothesis
                           ]
        
        cvc = 0
        while not self._should_terminate and cvc < len(functions_to_run):
            function = functions_to_run[cvc]
            function()
            cvc += 1
        
    def predict(self, time_to_predict, time = None, event = None, group_by = None, model = "kmf", raw = False):
        """
        Predicts the survival probability given time. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column 
        time_to_predict : float
            time value that correspons to how long a use has been in the company
        model : str
            model to run the prediction on. Can be either `kmf`, or `naf`
        raw : bool
            if True, then prints the float fraction, otherwise prints a string with percentage.
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set() == False:
            return
        
        time_variable = self.data_encoded[self.time]
        event_variable = self.data_encoded[self.event]
        if model == "kmf":
            kmf = self.__fit_kmf(time_variable, event_variable)
            if kmf:
                pred = kmf.predict(time_to_predict)
            
        elif model == "naf":
            naf = self.__fit_naf(time_variable, event_variable)
            if naf:
                pred = naf.predict(time_to_predict)
            
        if raw:
            return pred
        else:
            return f"{int(pred * 100)}%"
        
    def test_hypothesis(self, time = None, event = None, group_by = None):
        """
        Runs hypothesis testing on a group. 
        
        Parameters
        ----------
        event : str
            name of the event column 
        time : str
            name of the time column 
        group_by : str
            name of the group_by column
        """
        
        self.__set_all_variables(event, time, group_by)
        if self.__check_all_variables_set(include_group = True) == False:
            return
         
        print("Hypothesis Testing\n")    
        
        print(f"Group column: {self.group_by}\n")       
        print("H0: Survival curves are not different")
        print("H1: Survival curves are different\n")
        
        random_group_value = self.data_encoded[self.group_by].unique()[0]
        first = self.data_encoded[self.data_encoded[self.group_by] == random_group_value]
        second = self.data_encoded[self.data_encoded[self.group_by] != random_group_value]
        T = first[self.time]
        E = first[self.event]
        T1 = second [self.time]
        E1 = second[self.event]

        results = logrank_test(T, T1, event_observed_A = E, event_observed_B = E1)
        
        chi2 = results.summary["test_statistic"][0]
        p_value = results.summary["p"][0]
        
        print(f"Chi Squared: {chi2}")
        print(f"p-value: {p_value}\n")
        
        if p_value < 0.05:
            print("We reject H0 => Survival curves are indeed different")
        else: 
            print("We fail to reject H0 => Survival curves are not different")
    
    def __transform_categorical(self, data):
        
        data_encoded = data
        cat_code_dfs_dict = dict()

        for c in data:
            unique_vals = data_encoded[c].unique()

            if len(unique_vals) < 10:
                col_cat = self.data[c].astype("category")
                categories = list(col_cat.cat.categories)
                cat_dict = {key : value for key, value in enumerate(categories)}
                cat_code_dfs_dict[c] = cat_dict
                
                data_encoded[c] = col_cat.cat.codes
                
        return data_encoded, cat_code_dfs_dict
    
    def __set_all_variables(self, event, time, group_by):
        if event == None:
            event = self.event
        else:
            self.event = event
            
        if time == None:
            time = self.time
        else:
            self.time = time
            
        if group_by == None:
            group_by = self.group_by
        else:
            self.group_by = group_by
    
    def __check_all_variables_set(self, include_group = False):
        allSet = self.event != None and self.time != None and (self.group_by != None or not include_group)
        event_present = self.event in self.data.columns
        time_present = self.time in self.data.columns
        group_present = self.group_by in self.data.columns
        
        if not event_present:
            print("ERROR: Event column not specified or not found")
        
        if not time_present:
            print("ERROR: Time column not specified or not found")
        
        if include_group and not group_present:
            print("ERROR: Group column not specified or not found")
        
        return allSet and event_present and time_present and (group_present or not include_group)
    
    def __get_time_event(self, grouped_df):
        try:
            time_variable = grouped_df[self.time]
        except KeyError:
            print("ERROR: Time variable not found. Please check your input.")
            return None
        
        try:
            event_variable = grouped_df[self.event]
        except KeyError:
            print("ERROR: Time variable not found. Please check your input.")
            return None
        
        return (time_variable, event_variable)
    
    def __fit_kmf(self, time_variable, event_variable, label = None):
        try:
            kmf = KaplanMeierFitter()
            kmf.fit(time_variable, event_variable, label = label)
            return kmf
        except TypeError:
            print("Could not run fit function. Check the input columns.")
            self._should_terminate = True
            return None
        
    def __fit_naf(self, time_variable, event_variable, label = None):
        try:
            naf = NelsonAalenFitter()
            naf.fit(time_variable, event_variable, label = label)
            return naf
        except TypeError:
            print("Could not run fit function. Check the input columns.")
            self._should_terminate = True
            return None
            
    def __print_divider(self):
        print("\n" + "-" * 80)
        
    def __repr__(self):
        return f"""
    SurvivalAnalysis
        data : {self.data_file_path}
        """

    