#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    error = list((net_worths - predictions)**2)
    cleaned_data = zip(ages, net_worths, error) # return a list of tuple
    sorted_data = sorted(cleaned_data, key = lambda tup: tup[2])
    num_retain = int(len(sorted_data) * .9)
    cleaned_data = sorted_data[:num_retain]
    
    return cleaned_data


