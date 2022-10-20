lr = 6e-4
num_queries = 5
query_size = 10

mesh_alpha = 0.7

sorted_pool_list = []
bald_out_list = []
labeled_idx_list = []

T = 100
bald_method = 'MC_drop'

label_list = ['bald', 'var', 'bald_1', 'bald_2', 'softmax']

# figure init
fig, axs = plt.subplots(nrows=len(label_list), ncols=num_queries+1, figsize=(20, 16),sharex=True, sharey=True)

# reset dataset, model and optimizer
traindata.reset_mask()
model = MLP(drop_out=drop_out)
optimizer = optim.Adam(model.parameters(), lr = lr)

# plot initial 10 data points, first col in plot
xx, yy, grids_list = BALD_grid_viz(model, X_train, y_train, T = T)
xx_soft, yy_soft, softmax_out = softmax_grid(model, X_train, y_train)

for i, g in enumerate(grids_list):
    mesh = axs[i,0].pcolormesh(xx, yy, g, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
    axs[i,0].scatter(X_train[init_pool_idx,0], X_train[init_pool_idx,1], c= y_train[init_pool_idx])
    axs[i,0].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.2)
    axs[i,0].set_xlim(X_train[:,0].min(), X_train[:,0].max())
    axs[i,0].set_ylim(X_train[:,1].min(), X_train[:,1].max())
    
mesh = axs[4,0].pcolormesh(xx_soft, yy_soft, softmax_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
axs[4,0].scatter(X_train[plot_idx,0], X_train[plot_idx,1], c = y_train[plot_idx])
axs[4,0].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.2)
axs[4,0].set_xlim(X_train[:,0].min(), X_train[:,0].max())
axs[4,0].set_ylim(X_train[:,1].min(), X_train[:,1].max())


# train on initial pool
traindata.update_mask(init_pool_idx)
labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                    sampler=SubsetRandomSampler(init_pool_idx), shuffle = False)

model = train(model, labeled_loader, optimizer, device, num_epochs=num_epochs, plot = False, printout = False)

for j, query in enumerate(range(num_queries)):
    print(f'// Query {j+1:2d} of size {query_size}')
    
    # quering data points
    sample_idx, scores, sorted_pool, Xs = query_the_oracle(model, traindata, device, query_strategy='bald', bald_method =  bald_method, query_size=query_size, batch_size=batch_size, T = T)
    sorted_pool_list.append(sorted_pool)
    
    #print(f'...with BALD scores: \n {scores}')
    #print(f'...and coordinates : \n {Xs}')
    
    traindata.update_mask(sample_idx)
    labeled_idx = np.where(traindata.unlabeled_mask == 0)[0]
    labeled_idx_list.extend(sample_idx)
                
    # define a list for plotting the initial pool + the queried points
    plot_idx = init_pool_idx + labeled_idx_list    

            
    labeled_loader = DataLoader(traindata, batch_size=batch_size, num_workers=0,
                                    sampler=SubsetRandomSampler(labeled_idx), shuffle = False)
    
    
    
     # plot
    xx, yy, grids_list = BALD_grid_viz(model, X_train, y_train, T = T)
    xx_soft, yy_soft, softmax_out = softmax_grid(model, X_train, y_train)
    
    bald_out_list.append(grids_list[0])
    
    for k, g in enumerate(grids_list):
        mesh = axs[k,j+1].pcolormesh(xx, yy, g, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
        axs[k,j+1].scatter(X_train[plot_idx,0], X_train[plot_idx,1], c = y_train[plot_idx])
        axs[k,j+1].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.2)
        axs[k,j+1].set_xlim(X_train[:,0].min(), X_train[:,0].max())
        axs[k,j+1].set_ylim(X_train[:,1].min(), X_train[:,1].max())
    
    mesh = axs[4,j+1].pcolormesh(xx_soft, yy_soft, softmax_out, cmap=plt.cm.RdBu_r, alpha = mesh_alpha)
    axs[4,j+1].scatter(X_train[plot_idx,0], X_train[plot_idx,1], c = y_train[plot_idx])
    axs[4,j+1].scatter(X_train[:,0], X_train[:,1], c = y_train, alpha = 0.2)
    axs[4,j+1].set_xlim(X_train[:,0].min(), X_train[:,0].max())
    axs[4,j+1].set_ylim(X_train[:,1].min(), X_train[:,1].max())
    
    
    # train model
    optimizer = optim.Adam(model.parameters(), lr = lr)
    model = train(model, labeled_loader, optimizer, device, num_epochs=num_epochs, plot = False, printout = False)


fig.tight_layout()
fig.colorbar(mesh, ax=axs.ravel().tolist(), fraction=0.01, pad=0.01)

# set ylabels to strategy
for i, label in enumerate(label_list):
    axs[i,0].set_ylabel(label, fontsize=20)

# set xlabels to number of sampled data points.
ns = np.linspace(0, query_size*num_queries, num_queries+1, dtype=int)

for i, n in enumerate(ns):
    axs[len(label_list)-1,i].set_xlabel(f'n={int(init_pool_size+n)}', fontsize=18)

plt.savefig(FIGURE_PATH+'bald_viz_3.png')
plt.show()