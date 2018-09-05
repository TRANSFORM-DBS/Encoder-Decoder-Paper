%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training Phase (Ishita or Angelique might help for this step)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all
x_min = -2;
x_max =  2;
sample= 2000;
Xs    = linspace(x_min,x_max,sample);

% call this on learning data
% define file name and state variable that being estimated
file_name        = 'MG79_decoder_training.mat';
ModelSetting.pVal             = (0.001)^1;
ModelSetting.SelMode          = 4;
ModelSetting.NoStateSamples   = 5;
ModelSetting.which_state      = 1;
ModelSetting.training_section = 2;  % 1: alternate points, 2: first half, 3: last half  
ModelSetting.Xs = Xs;  
ModelSetting.SL = 10;  % significance leveles (10,5,1)
ay_neural_decoder_training(file_name,ModelSetting);

%%%---------------------------------------------------------> ay_map_to_sim();
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Real-Time Procdure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% call this onceh
% load the model
load('Model');
% build state-transition distribution
TransP= ones(length(Xs),length(Xs));
for i=1:length(Xs)
    TransP(i,:)=pdf('normal',Xs(i),sParam.a*Xs,sqrt(sParam.sv));
end
%% load test file
load(file_name);
temp = cell2mat(XPos');
XM   = temp(ModelSetting.which_state,:);
temp = cell2mat(SPos');
XS =temp(1,1:2:length(temp));
no_feature = size(Y,2);
%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Example for Real Time Implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% we keep previous posterior - initialize XPre
XPre = ones(1,size(TransP,1));
% we might use previous point of feature
Yprv = zeros(1,no_feature);
% this is the hypothetical real-time loop
% I am keeping mean of estimate here
MEAN = [];
LOW  = [];
HI   = [];
lMap = [];
fMap = [];
valid=dValid(:,2)& dValid(:,3);
%valid=dValid(:,1);
id=find(valid==1);

for n=1:length(XM)
    
    % load Yk
    Yk = Y(n,:);
    %if(sum(isnan(Yk))>0)
    % decoder model 
    [XPos,CurEstimate,Xll] = ay_one_step_decoder(data_type,eParam,XPre,TransP,Xs,valid,Yk,Yprv);
    lMap = [lMap;Xll];
    fMap = [fMap;XPos];
    % next step
    XPre = XPos;
    Yprv = Yk;
    %end
    % result
    MEAN = [MEAN;CurEstimate.Mean];
    LOW  = [LOW;CurEstimate.Bound(1)];
    HI   = [HI;CurEstimate.Bound(2)];
end

shadedErrorBar(1:length(XM),XM,2.*sqrt(XS),{'b','markerfacecolor','b','linewidth',1},1);hold on
shadedErrorBar(1:length(MEAN),MEAN,[HI-MEAN,MEAN-LOW],{'r','markerfacecolor','r','linewidth',1},1);
box off; xlim([0 length(XM)])

