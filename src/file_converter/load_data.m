
%% Create in Wednesday Decmber 2022 4:26pm
%% The function was created with helper code from the toolbox along with the dataset 

function [data, label] = load_data(path,subject,session,task_type,sig_type)


file_name = dir([path,'/pr_dataset/subject',subject,'_session',num2str(session),'/',task_type,'_',sig_type,'*.dat']);

for i=1:length(file_name)
    Fname = [path,'/pr_dataset/subject',subject,'_session',num2str(session),'/',task_type,'_',sig_type,'_sample',num2str(i),'.dat'];         % Name with path.
    fid = fopen(Fname,'r','n');            % Open for reading.
    if fid<0, error(['Failed to open: ' Fname]); end
    data_tmp = fread(fid, [256,inf], 'int16');  % Read.
    fclose(fid);
    data_tmp=data_tmp;
    
    Fname = [path,'/pr_dataset/subject',subject,'_session',num2str(session),'/',task_type,'_',sig_type,'_sample',num2str(i),'.hea'];         % Name with path.
    head_info=textread(Fname,'%s');
    idx = find( strcmp( head_info , [task_type,'_',sig_type,'_sample',num2str(i),'.dat'] ));
    
    for j=1:length(idx)
        str_tmp=head_info(idx(j)+2);
        idx2=strfind( str_tmp , '(' );
        gain=str2num(str_tmp{1,1}(1:(idx2{1,1}-1)));
        idx3=strfind( str_tmp , ')' );
        baseline=str2num(str_tmp{1,1}((idx2{1,1}+1):(idx3{1,1}-1)));
        
        data_tmp(:,j)=(data_tmp(:,j)-baseline)/gain;
    end
    data{1,i} = data_tmp;
end


label = importdata([path,'/pr_dataset/subject',subject,'_session',num2str(session),'/label_',task_type,'.txt']);


end

