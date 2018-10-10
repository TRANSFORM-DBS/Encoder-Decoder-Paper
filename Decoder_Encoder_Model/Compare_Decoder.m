clear all

file_name = 'MG112_features_extended.mat';
x_min = -2;
x_max =  2;
sample= 2000;
Xs    = linspace(x_min,x_max,sample);

load(file_name);

ModelName = 'Decider_Model_MG112_2018_2_22_15_14_58.mat';
load(ModelName);
ind=find(dValid(:,1)==1); 

temp = cell2mat(XPos');
XM   = temp(ModelSetting.which_state,:);
no_feature = size(Y,2);

%%---------------------------------------------
% build state-transition distribution
TransP = ones(length(Xs),length(Xs));
for i=1:length(Xs)
    TransP(i,:)=pdf('normal',Xs(i),sParam.a*Xs,sqrt(sParam.sv));
end

% Decoding using Ali's code
XPre = ones(1,size(TransP,1));
Yprv = zeros(1,no_feature);
MEAN=[]; LOW=[]; HI=[]; lMap=[]; fMap=[];
for n=1:size(Y,1)
    % load Yk
    Yk = Y(n,:);
    % decoder model 
    [XPos,CurEstimate,Xll] = ay_one_step_decoder(data_type,eParam,XPre,TransP,Xs,dValid(:,1),Yk,Yprv);
    lMap = [lMap;Xll];
    fMap = [fMap;XPos];
    % next step
    XPre = XPos;
    Yprv = Yk;
    % result
    MEAN = [MEAN;CurEstimate.Mean];
    LOW  = [LOW; CurEstimate.Bound(1)];
    HI   = [HI;  CurEstimate.Bound(2)];
end
XM1=MEAN;
L1=LOW;
H1=HI;

% Decoding using Rina's code
Param_W=[];
Param_Dispersion=[];
Param_EncModel_XPow=[];

for iFeat=1:length(eParam)
    Param_W(iFeat,:) = eParam{iFeat}.W;
    Param_Dispersion(iFeat,:) = eParam{iFeat}.Dispersion;
    Param_EncModel_XPow(iFeat,:) = eParam{iFeat}.EncModel.XPow;
end

XPre = ones(1,size(TransP,1));
Yprv = zeros(1,no_feature);
MEAN=[]; LOW=[]; HI=[]; lMap=[]; fMap=[];
for n=1:size(Y,1)
    % load Yk
    Yk = Y(n,:);
    % decoder model 
    [XPos,CurEstimate,Xll] = ay_one_step_decoder_realTime(data_type, Param_W, Param_Dispersion, Param_EncModel_XPow, XPre, TransP, Xs, dValid(:,1), Yk, Yprv);
    lMap = [lMap;Xll];
    fMap = [fMap;XPos];
    % next step
    XPre = XPos;
    Yprv = Yk;
    % result
    MEAN = [MEAN;CurEstimate.Mean];
    LOW  = [LOW; CurEstimate.Bound(1)];
    HI   = [HI;  CurEstimate.Bound(2)];
end
XM2=MEAN;
L2=LOW;
H2=HI;

plot(XM1,'k','linewidth',2);hold on
plot(XM2,'r--','linewidth',2);
plot(L1,'b','linewidth',1);plot(H1,'b','linewidth',1);
plot(L2,'g--','linewidth',1);plot(H2,'g--','linewidth',1);
plot(XM,'b','linewidth',2)
box off
legend('Mean from Ali code','Mean from Rina code')