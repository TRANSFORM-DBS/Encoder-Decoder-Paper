% now we are part of github for real!
clear all
EXAMPLE = 1
if EXAMPLE==1
    file_name = 'Example_A_features_encoder.mat';
    ModelSetting.pName = 'EXAMPLE_A';
else
    file_name = 'Example_B_features_encoder.mat';
    ModelSetting.pName = 'EXAMPLE_B';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Training Phase (Ishita or Angelique might help for this step)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_min = -2;
x_max =  2;
sample= 2000;
Xs    = linspace(x_min,x_max,sample);

%%--------------------------------------------------------------
% call this on learning data
% define file name and state variable that being estimated
ModelSetting.pVal             = 0.01;       % 0.05, 0.01, 0.001
ModelSetting.SelMode          = 6;          % 6 or 7
ModelSetting.NoStateSamples   = 1000;
ModelSetting.which_state      = 1;
ModelSetting.Xs = Xs;  

% Load file containing neural features and state values
load(file_name);
temp = cell2mat(XPos');
XM   = temp(ModelSetting.which_state,:);
no_feature = size(Y,2);

 L=length(XM);

TrainInd=1:L/2;
TestInd=setdiff(1:L,TrainInd);

ModelName=ay_neural_encoder_training(file_name,ModelSetting,TrainInd);

%%----------------------------------------------------
% Real-Time Procdure
% load the model
load(ModelName);
ind=find(dValid(:,1)==1);  % If there are too many features that were passed by f-test , can reduce pvalue and run again
XM=XM(ValidId);
%%---------------------------------------------
% build state-transition distribution
TransP = ones(length(Xs),length(Xs));
for i=1:length(Xs)
    TransP(i,:)=pdf('normal',Xs(i),sParam.a*Xs,sqrt(sParam.sv));
end

%%-----------------------------------------------

figure(1)
plot(dValid(:,4),'LineWidth',2);
hold on
ind =find(dValid(:,1));
plot(ind,dValid(ind,4),'o','LineWidth',2);
box off
title('R^2 and Valid Features')
xlabel('Feature Index')
ylabel('R^2')

figure(2)
[~,m_ind] = max(dValid(:,4));
[~,c_ind] = min(dValid(:,4));
tY = mean(eParam{m_ind}.Y);
fY = Y(TrainInd,m_ind);
subplot(3,1,1)
plot(fY,'LineWidth',2);
hold on
plot(tY,'LineWidth',2);
box off
title(['model ' eParam{m_ind}.RefModel  ', slope(x)=' num2str(eParam{m_ind}.W(2))])
legend('Feature','Prediction')
axis tight
subplot(3,1,2)
plot(TrainInd,XM(TrainInd),'LineWidth',2);
xlabel('Training Index')
ylabel('X');box off
axis tight
subplot(3,1,3)
plot(XM(TrainInd),Y(TrainInd,m_ind),'*');hold on;plot(XM(TrainInd),Y(TrainInd,c_ind),'o');
xlabel('X')
ylabel('Z')
box off
axis tight
title('Scatter Plot')

%%------------------------------------
%This is the 2nd step for shrinking neural features

TProb = ay_individual_decoder(data_type,eParam,Xs,dValid(:,1),Y);
XProb = TProb;

% Subset feature given Training/Testing/Full Dataset
ValidInd=TrainInd; % Other options can be ValidInd=TestInd;  ValidInd=1:L;

for f=1:length(TProb)
    if  TProb{f}.valid
        TProb{f}.prb=TProb{f}.prb(ValidInd,:);
    end
end
[rmse_ind,rmse_curve,optim_curve,winner_list] = ay_sort_decoder_sub(TProb,Xs,dValid(:,1),SampleX(:,ValidInd));
[~,mid]=min(rmse_curve);
opt_id=rmse_ind{mid};

figure(3)
subplot(2,1,1)
plot(rmse_curve,'LineWidth',2);hold on
title('Model Selection Given Training Data');
axis tight
subplot(2,1,2)
plot(optim_curve(mid,:),'LineWidth',2);
hold on
plot(XM(ValidInd),'LineWidth',2);
hold off
ylabel('Y')
xlabel('ValidInd')
axis tight
disp('Optimal Feature Set Given Training Data')


%% Alternative choice for model selection
% First L features with highest r2
L=5;
[sval,sid]=sort(dValid(:,4),'descend');
opt_id=sid(1:L);

% Visualizing what the mean decoded state looks like with the reduced
% features v all

tdValid=zeros(length(dValid(:,1)),1);
tdValid(opt_id)=1;
%%---------------------------------------------------
% we keep previous posterior - initialize XPre
XPre = ones(1,size(TransP,1));
%XPre = pdf('normal',Xs,XPos{1}(1),10.*sqrt(SPos{1}(1,1))); 
% we might use previous point of feature
Yprv = zeros(1,no_feature);
% this is the hypothetical real-time loop
% I am keeping mean of estimate here
MEAN=[]; LOW=[]; HI=[]; lMap=[]; fMap=[];
for n=1:size(Y,1)
    % load Yk
    Yk = Y(n,:);
    % decoder model 
    % Using subset of features
    if (~isnan(Yk))
    [XPos,CurEstimate,Xll] = ay_one_step_decoder(data_type,eParam,XPre,TransP,Xs,tdValid,Yk,Yprv);
    lMap = [lMap;Xll];
    fMap = [fMap;XPos];
    % next step
    XPre = XPos;
    Yprv = Yk;
    % result
    MEAN = [MEAN;CurEstimate.Mean];
    LOW  = [LOW; CurEstimate.Bound(1)];
    HI   = [HI;  CurEstimate.Bound(2)];
    else
        continue
    end
end

%%--------------------------------------------
% plot figure, State Mean plus Mean+/-Std Estimate
figure(5)
plot(XM,'b','LineWidth',2);hold on;
plot(MEAN,'r','LineWidth',2);hold on
plot(HI,'r--');
plot(LOW,'r--');
legend('behavior estimate','neural estimate');
title('State Estimate with features using test data');
box off

figure(6)
imagesc(1:length(XM),Xs,(lMap'));
hold on
plot(1:length(XM),XM,'w.');
title('Likelihood');
xlabel('Trial Index')
ylabel('State Estimate')
hold off

% plot figure, filter estimate plus State Mean
figure(7)
imagesc(1:length(XM),Xs,(fMap'));
hold on
plot(1:length(XM),XM,'w.');
title('Filter Estimate');
xlabel('Trial Index')
ylabel('State Estimate')
hold off
axis tight

