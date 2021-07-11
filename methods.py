def corralation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    return df
def cross_validation(X, Y, model, num_splits=5):
  # split data -> train1_x, test1_x, train1_y, test1_y = model_selection.train_test_split(X, Y, random_state = 0)
  # alternative version -> cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )
  results = model_selection.cross_val_score(model, X, Y, cv  = num_splits)
  return results
