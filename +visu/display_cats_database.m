function display_cats_database(database)

    %-- Display part of the training database
    m_train = size(database.X_train,2);
    m_test = size(database.X_test,2);

    rand_indices = randperm(m_train);
    sel = database.X_train(:,rand_indices(1:64));
    visu.display_database(sel,database.num_px,database.num_px,'Training samples');

    %-- Display testing database
    rand_indices = randperm(m_test);
    sel = database.X_test(:,rand_indices);
    visu.display_database(sel,database.num_px,database.num_px,'Testing samples');

end
