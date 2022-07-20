#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograms(df, columns):
    """ Plots multiple histograms of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.histplot(data=df[col], bins=10)

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_boxplots(df, columns):
    """ Plots multiple boxplots of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[col])

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_variable_pairs(df, numerics, categoricals, targets, sample_amt):
    """ Plots pairwise relationships between numeric variables in df along with regression line for each pair. Uses categoricals for hue."""
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    # Checks if a sample amount was inputted
    if sample_amt:
        df = df.sample(sample_amt)
    # Checks if any categorical variables were given to determine how to set the lmplot regression line parameters
    if len(categoricals)==0:
        categoricals = [None]
        # Setting to red makes it easier to see against the default color
        line_kws = {'lw':4, 'color':'red'}
    else:
        line_kws = {'lw':4}
    for cat in categoricals:    
        for col in numerics:
            for y in targets:
                if y == col:
                    continue
                sns.lmplot(data = df, 
                           x=col, 
                           y=y, 
                           hue=cat, 
                           palette='Set1',
                           scatter_kws={"alpha":0.2, 's':10}, 
                           line_kws=line_kws,
                           ci = None)
            

def plot_categorical_and_continuous_vars(df, categorical, continuous, sample_amt):
    """ Accepts dataframe and lists of categorical and continuous variables and outputs plots to visualize the variables"""
    sns.set(font_scale=1.2)
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    if sample_amt:
        df = df.sample(sample_amt)
    # Outputs 3 plots showing high level summary of the inputted data
    for num in continuous:
        for cat in categorical:
            _, ax = plt.subplots(1,3,figsize=(20,8))
            print(f'Generating plots {num} by {cat}')
            p = sns.stripplot(data = df, x=cat, y=num, ax=ax[0], s=1)
            p.axhline(df[num].mean())
            p = sns.boxplot(data = df, x=cat, y = num, ax=ax[1])
            p.axhline(df[num].mean())
            p = sns.violinplot(data = df, x=cat, y=num, hue = cat, ax=ax[2])
            p.axhline(df[num].mean())
            plt.suptitle(f'{num} by {cat}', fontsize = 18)
            plt.show()

