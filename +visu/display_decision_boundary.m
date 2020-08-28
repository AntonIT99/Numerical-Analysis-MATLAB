function display_decision_boundary(database,parameters)

    %-- Extract points that belong to each class
    X_train_c0 = database.X_train(:,database.Y_train==0);
    X_train_c1 = database.X_train(:,database.Y_train==1);

    %-- Create a X_mesh grid based on the position of the input points
    xmin = min(database.X_train(1,:))-0.5;
    xmax = max(database.X_train(1,:))+0.5;
    ymin = min(database.X_train(2,:))-0.5;
    ymax = max(database.X_train(2,:))+0.5;
    h = 0.01;
    xvec = xmin:h:xmax;
    yvec = ymin:h:ymax;
    [xx,yy] = meshgrid(xvec,yvec);
    X_mesh = [xx(:)';yy(:)'];
    
    %-- For each node of the mesh, estimate the prediction based on the learned parameters
    z = L_layers_nn.predict(parameters, X_mesh);
    zz = reshape(z,size(xx));
    
    %-- Display the corresponding result
    figure; imagesc(xvec,yvec,zz); axis image; colormap(jet);
    hold on; plot(X_train_c0(1,:),X_train_c0(2,:),'ob','linewidth',2);
    hold on; plot(X_train_c1(1,:),X_train_c1(2,:),'or','linewidth',2);

end
