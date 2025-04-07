%% Create in Wednesday Decmber 2022 4:26pm
%% The function was created with helper code from the toolbox along with the dataset

clear all;
close all;

subject = {'01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20'};
session = 1;
task_type = 'maintenance';
signal_type =  'preprocess';  %'raw'; %'preprocess';

load_path = [''];
save_path = [''];


for i=1:length(subject)

    save_data(load_path, save_path, subject{1,i}, session, task_type,signal_type);
end

