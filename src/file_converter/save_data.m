%% Create in Wednesday Decmber 2022 4:26pm
%% The function was created with helper code from the toolbox along with the dataset 

function [] = save_data(load_path, save_path, subject, session, task_type,sig_type)

if strcmp(sig_type,'preprocess')
join_save_path = [save_path, '\preprocess', '\subject_', subject,'\', '\session_', num2str(session)];

elseif strcmp(sig_type, 'raw')
    join_save_path = [save_path, '\raw', '\subject_', subject,'\', '\session_', num2str(session)];
else
    disp('Use the right signal type');
end

data_f_name = [join_save_path, '\','data.mat'];
label_f_name = [join_save_path, '\', 'label.mat'];



[data, label] = load_data(load_path,subject,session,task_type,sig_type);

if ~exist(join_save_path, 'dir')
    mkdir(join_save_path);
end

save(data_f_name, 'data');
save(label_f_name, 'label');
end

