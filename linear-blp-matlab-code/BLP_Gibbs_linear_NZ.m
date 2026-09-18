%BLP_Gibbs_linear_NZ.m
%Code to estimate Bayesian Local Projections (BLP) using Gibbs sampling
%with New Zealand data
%
%If you use this BLP method and code, please cite "Diminishing Monetary
%Potency under Fiscal Dominance: A Bayesian Local Projections Analysis" by
%Naveed Javed and James Morley, September 2026, CAMA WP
%
%Code allows for multiple shocks/instruments
%Allows Autocorrelation Correction (AC) using controls, SUR inferences across horizons and possibly endogenous variables
%Allows for BLP-IV estimation using a 2SLS Control Function (CF) approach
%
%Bayesian posterior bands are equal-tailed
%
%By including errors from all equations as controls for serial correlation
%the system becomes SUR (same controls for a given horizon and errors at
%different horizons should be uncorrelated. However, better to be
%parsimonious in controls
%
%Allows for Covid dummies in 2020Q2 and 2020Q3 to handle outliers in real GDP
%Allows for ZLB dummies from 2020Q2 to 2021Q3 to handle ZLB on OCR
%Allow for multiple instruments for OCR
%Need to orthogonalize 1st stage residuals to make 2SLS and 2SLSCF equivalent
%Allows block exogenous controls such as US variables for an SOE
%
%Code updated by James Morley on 1 September 2026

clear all
close all
clc

addpath([pwd '/_func/']) %mac

seed=9999;
rng(seed); %fix seed for replicability


%% IMPORT and SETUP DATA

data = xlsread('data_NZ.xlsx','Main'); 
time = data(:,1);
t1 = find(time==2000.25);
%t1 = find(time==2000.5); %uncomment if using GSS shocks
%t2 = find(time==2019.75);
%t2 = find(time==2025.75);
t2 = find(time==2025.5); %drop last observation because using lagged GDP for debt-to-GDP ratio

data = data(t1:t2,:);

endogenous_variables = [data(:,2),data(:,5:7)]; %OCR LnGDP Inflation LnTWI(real)
%controls = data(:,x); %Controls to also include on the RHS
controls = 0; %uncomment to not include extra controls

block_exogenous_controls = [data(:,17),data(:,18),data(:,20)]; %Block exogenous controls that have contemporaneous values included along with the lags
%block_exogenous_controls = 0; %uncomment to not include extra block exogenous controls

shock_series = data(:,11); %Romer-Romer type monetary policy shocks 
%shock_series = 0.01*data(:,12); %GSS type monetary policy shocks
%shock_series = [data(:,11),0.01*data(:,12)]; %Two instruments 

dummies1 = data(:,21:26); %separate dummies for 2020Q2 to 2021Q3 for ZLB
dummies1 = zeros(length(data),6); %uncomment to remove ZLB dummies
dummies2 = data(:,13:14); %separate dummies for 2020Q2 and 2020Q3 for Covid
%dummies2 = zeros(length(data),2); %uncomment to remove Covid dummies

norm_sign = ones(size(shock_series,2),1); %1 normalizes as a contractionary monetary policy shock

shock_series_positive = shock_series.*(shock_series>0);
shock_series_negative = shock_series.*(shock_series<0); 
%shock_series = shock_series_positive; %uncomment to use positive shocks only
%shock_series = shock_series_negative; %uncomment to use negative shocks only

%comment or uncomment next two lines to consider symmetry or asymmetry
%norm_sign = [1;-1]; %uncomment to multiply second shock by -1 to normalize as an expansionary shock
%shock_series = [norm_sign(1)*shock_series_positive,norm_sign(2)*shock_series_negative]; %uncomment to consider both shock series jointly

T = length(endogenous_variables);
N = size(endogenous_variables,2); %number of endogenous variables
k=size(shock_series,2); %number of shocks series
l=k; %number of shocks/instrumented variables
%l=1;  %uncomment if IV with multiple instruments but just one variable (OCR) being instrumented
n_x=0; %set to 0 to denote no regime switching for sign restrictions for this code


%% GLOBAL SETTINGS
p = 4; %number of lags to include in controls
h_max = 20;   %maximum horizon for IRFs
iv = 0; %set to 1 to consider LP-IV; Note: counterfactuals for IV will not make sense given normalized IRFs
orthogonalize = 1; %orthogonalize first stage errors for IV=1 and l>1
normalize = 1; %set to 1 to rescale IRFs by initial impact on first outcome variable (e.g., OCR) if not using IV
c_level = 0.68;%0.95; %coverage for confidence bands
ld = 0; %set to 1 to consider long differences specification
ps = 0; %set to 1 to use Piger-Stockwell controls of first differences of response variable 
p_max = p; %useful to store set p when considering long differences


%% AUTOCORRELATION CORRECTION
neweywest = 1; %set to 1 to use Newey-West standard errors for Classical LP, OLS otherwise
ac = 1; %set to 1 to use univariate controls for serial correlation for BLP, 2 to use lagged errors (or projection errors) for all N
    control_truncation=1;%h_max;  %if set to h_max, no truncation for ac=1. Otherwise, the number of lags given
leads = 1; %set to 1 to control for leads and lags of MP shock or -1 to consider AR(p) correction 
if leads==-1
[~,shock_series_resids]=ARp_fcn(shock_series,p,0,0,0);
shock_series=shock_series_resids;
end


%% INFORMATIVE PRIORS FOR BLP ANALYSIS
minnesota_lr = 0; %set to 1 to put Minnesota like prior on IRFs at longer horizons
    lambda_irf = 30;%0.75;

smoothing = 1; %set to 1 to estimate using changes in IRFs to impose smoothing shape prior, overrides minnesota_lr
    lambda_smooth = 8;4;   %set low to smooth IRFs   
    gamma_lr = 0.99;0.975;         %set low to place prior mean that IRFs are zero in the long run, set to 1 for prior mean for diff is zero 
    peak = 6;4;   %set prior on peak impact period of shock
    base = 12;

minnesota_lags = 0; %set to 1 to put Minnesota like prior on coefficients on lagged variables, can consider larger p
    lambda_lags = 1;%0.2;

minnesota_variance = 1; %set to 1 to put Minnesota like prior on variance of residual
    lambda_var = 50; %9 (50) implies prior standard deviation of about 1 (0.23) if target mode is 1
    horizon_expand = 0; %set to 1 if target mode of variance increases with horizon

minnesota_controls=1; %set to 1 to put Minnesota like prior on controls if AC=1
    lambda_controls=(1/3); %set low to shrink to prior mean over long horizons, 0.25 has more precise long horizon estimates, 0.5 removes all serial correlation
    mean_decay=0.75;       %set to 1 to have prior mean of 1, less than one to decay over horizons

prior_iv=1; %set to 1 to impose a prior on first-stage instrument relationship to endogenous variable
    iv_prior_mean=1;
    iv_prior_var=0.2^2;

impact_prior=1; %set to 1 to impose prior of norm_sign on initial effects of shock on policy variable
    lambda_impact = 0.2;

sign_lag=0; %set to horizon if using sign restrictions, -1 for no restrictions; apply to variables in GEN_BETA.m  


%%  Classical LP using OLS or IV

if ps== 0
[irfs, ses, prmtr_point_store, ~] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads);    
else
[irfs, ses, prmtr_point_store, ~] = CLP_fcn_PigerStockwell(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads);    
end

if normalize == 1  %rescale irfs to be for unit shock in terms of impact on OCR
%rescale irfs to be for unit shock in terms of impact on OCR
%rescale irf SEs given unit shock in terms of OCR, noting SE for contemporaneous impact on OCR is zero
normalize_factor=[];
for i=1:l
    normalize_factor=[normalize_factor;irfs(1,i)];
end
for j=1:N
    for i=1:l
            irfs(:,(j-1)*l+i)=norm_sign(i)*irfs(:,(j-1)*l+i)./normalize_factor(i);
            ses(:,(j-1)*l+i)=ses(:,(j-1)*l+i)./abs(normalize_factor(i));
            ses(1,i)=0;
    end
end
end

%Construct confidence bands
irfs_lower=irfs-norminv(1-(1-c_level)/2)*ses;
irfs_upper=irfs+norminv(1-(1-c_level)/2)*ses;


%% Bayesian LP using Gibbs

%Preliminaries

if smoothing == 1 %subtract off level effect of MP shock to estimate change in parameter for initial values in sampler 
if ps == 0
    [~, ~, prmtr_point_store, ~] = CLP_diff_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads);
else
    [~, ~, prmtr_point_store, ~] = CLP_diff_fcn_PigerStockwell(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads);
end
end

if iv == 1 %save 1st stage estimates from 2SLS to start up sampler for IV case
    [~, ~, ~, b1] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p_max,neweywest,ld,leads);
    prmtr1_point_store = b1(:,1:l)';
end

[sige2,~]=ARp_fcn(endogenous_variables,4,dummies1,dummies2,ld); %Estimate OLS residual variances for AR(4) model for Minnesota priors

%Setup storage, number of simulations, and data

fnl_mat=[];
fnl_mat2=[];

%===== Number of Simulations ===========
N0 = 1000;       %NUMBER OF DRAWS TO LEAVE OUT
MM = 3000;       %NUMBER OF DRAWS AFTER BURN-IN
CAPN = N0 + MM;  %TOTAL NUMBER OF DRAWS

p = p_max; %reset back to original p if changed for long differences
      
prmtr_lag_store=zeros(N*(h_max+1),size(prmtr_point_store,2));
if iv == 1
prmtr_lag_store=[prmtr_lag_store,zeros(N*(h_max+1),l)];   
prmtr1_lag_store=zeros(l,size(prmtr1_point_store,2));
end
if ac==1
prmtr_lag_store=[prmtr_lag_store,zeros(N*(h_max+1),control_truncation)];
end
if ac==2
prmtr_lag_store=[prmtr_lag_store,zeros(N*(h_max+1),1)];
end
if smoothing==1
prmtr_level_store=prmtr_lag_store;
end
if leads==1
    leads_lag_store=zeros(N*(h_max+1)*k,p);
end
if sum(sum(dummies1))>0 && i==1
    dummies1_lag_store=zeros(h_max+1,size(dummies1,2));
end
if sum(sum(dummies2))>0 && i==2
    dummies2_lag_store=zeros(h_max+1,size(dummies2,2));
end
irfmm=[];
corrmm=[];
corr5mm=[];
corrhmm=[];
if smoothing == 1
irfdiffmm=[];
end

if ac == 2
errors_g=zeros(T,N*h_max);  %storage of projection errors from previous iteration
end

g=1;
while g <= CAPN     
g   

x_other_lags=[]; % to store the lags of other variables (not the response variable) in levels following Piger and Stockwell(2025)

irfg=[];  %storage for irf across horizons at a given draw
corrhg=[];
corrg=[];
corr5g=[];

if smoothing == 1
irfdiffg=[];  %storage for change in irf across horizons at a given draw
end
if ac == 1
errors_g=[];  %storage for errors from lower horizons to control for serial correlation and generate SUR inferences across horizons
end


h=0;  %horizon for LP@  
while h<=h_max  

errors_hg=[];

i=1;
while i<=N

p = p_max; %reset back to original p if changed for long differences    

if ld == 1
    p=p_max+1;
y=endogenous_variables(p+1+h:T,i)-endogenous_variables(p+1-1:T-h-1,i);
else
y=endogenous_variables(p+1+h:T,i);
end    
    
if iv == 1
    
instrumented_variables = norm_sign(1)*endogenous_variables(:,1);   

if l>1
instrumented_variables = [];    
for j=1:k
instrumented_variables = [instrumented_variables,norm_sign(j)*endogenous_variables(:,1).*(shock_series(:,j)>0)];
end
end  

x=[instrumented_variables(p+1:T-h,:),ones(T-p-h,1),(1:1:T-p-h)'];   %OCR, constant, time trend
if block_exogenous_controls ~= 0
    x=[x,block_exogenous_controls(p+1:T-h,:)];
    if ld ==1 && ps == 0
    x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
    end
end
    j=1;
    while j <= p_max
    if ps == 0
        x=[x,endogenous_variables(p+1-j:T-h-j,:)];  %lags of OCR, output, and inflation
    end
    if ld == 1
        if ps == 1
            x=[x,endogenous_variables(p+1-j:T-h-j,i)-endogenous_variables(p-j:T-h-1-j,i)]; % for first difference of response variables
        else
            x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,:)-endogenous_variables(p-j:T-h-1-j,:);
        end
    end
if ps == 1
        if i==1
            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,i+1:N)]; % for OCR, includes levels of all variables except OCR
        else
            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,1:i-1) endogenous_variables(p+1-j:T-h-j,i+1:N)];% includes levels of all variables except the response variable (other than OCR)
        end
end        
        if controls ~= 0
        x=[x,controls(p+1-j:T-h-j,:)];
        if ld == 1 && ps == 0
        x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);    
        end
    end
    if block_exogenous_controls ~= 0
        x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
        if ld == 1 && ps == 0
        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);    
        end
    end 
    j=j+1;
    end      

    if ps == 1
        x=[x,x_other_lags];
        x_other_lags=[];
    end
    

x_hat=[x,zeros(length(x),l)];

z=[shock_series(p+1:T-h,:),x(:,l+1:size(x,2))];

b = size(x_hat,2);

if ac == 1 

if h>0
    j=0;
    if h>control_truncation
        j=h-control_truncation;
    end
    while j<=h-1
        x_hat=[x_hat,errors_g(p+1+j:T-h+j,N*j+i)];  %add in errors for lower horizons to control serial correlation
    j=j+1;
    end    
end 

end

if ac == 2 

if h>0
   x_hat=[x_hat,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];  %add in multivariate errors for lower horizon to control serial correlation   
end 

end

n_xnd=size(x_hat,2); %number of variables excluding dummies

z11=z;

if sum(sum(dummies1))>0
    z11=[z11,dummies1(p+1+h:T,:)];
end

if leads==1
    j=1;
    while j <= min(p_max,h)
    x_hat=[x_hat,shock_series(p+1+j:T-h+j,:)]; %leads of MP shocks
    j=j+1;
    end 
end

if sum(sum(dummies1))>0 && i==1
    x_hat=[x_hat,dummies1(p+1+h:T,:)];
    z=[z,dummies1(p+1+h:T,:)];
end

if sum(sum(dummies2))>0 && i==2
    x_hat=[x_hat,dummies2(p+1+h:T,:)];
    z=[z,dummies2(p+1+h:T,:)];
end

if smoothing == 1
if h>0
        y=y-instrumented_variables(p+1:T-h,:)*prmtr_level_store((h-1)*N+i,1:l)'; %subtract off level effect of MP shock to estimate change in parameter@ 
end
end

else

x=[shock_series(p+1:T-h,1:k),ones(T-p-h,1),(1:1:T-p-h)'];   %mp shock, constant, time trend
if block_exogenous_controls ~= 0
    x=[x,block_exogenous_controls(p+1:T-h,:)];
    if ld ==1 && ps == 0
    x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
    end
end
    j=1;
    while j <= p_max
        if ps == 0
             x=[x,endogenous_variables(p+1-j:T-h-j,:)];  %lags of OCR, output, inflation, TWI
        end
        if ld == 1
            if ps == 1
                x=[x,endogenous_variables(p+1-j:T-h-j,i)-endogenous_variables(p-j:T-h-1-j,i)]; % for first difference of response variables
            else
                x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,:)-endogenous_variables(p-j:T-h-1-j,:);    
            end
        end
if ps == 1
        if i == 1
            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,i+1:N)]; % for OCR, includes levels of all variables except OCR
        else
            x_other_lags = [x_other_lags endogenous_variables(p+1-j:T-h-j,1:i-1) endogenous_variables(p+1-j:T-h-j,i+1:N)];% includes levels of all variables except the response variable (other than OCR)
        end
end        
        if controls ~= 0
        x=[x,controls(p+1-j:T-h-j,:)];
        if ld == 1 && ps == 0
        x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);    
        end
    end
    if block_exogenous_controls ~= 0
        x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
        if ld == 1 && ps == 0
        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);    
        end
    end 
    if leads==1
        x=[x,shock_series(p+1-j:T-h-j,:)];  
    end
    j=j+1;
    end      

    if ps == 1
        x=[x,x_other_lags];
        x_other_lags=[];
    end

b=size(x,2);

if ac == 1 

if h>0
    j=0;
    if h>control_truncation
        j=h-control_truncation;
    end
    while j<=h-1
        x=[x,errors_g(p+1+j:T-h+j,N*j+i)];  %add in errors for lower horizons to control serial correlation
    j=j+1;
    end    
end 

end

if ac == 2 

if h>0
   x=[x,errors_g(p+h:T-1,N*(h-1)+1:N*(h-1)+N)];  %add in multivariate errors for lower horizon to control serial correlation   
end 

end

n_xnd=size(x,2); %number of variables excluding dummies

if leads==1
    j=1;
    while j <= min(p_max,h)
    x=[x,shock_series(p+1+j:T-h+j,:)]; %leads of MP shocks
    j=j+1;
    end 
end

if sum(sum(dummies1))>0 && i==1
    x=[x,dummies1(p+1+h:T,:)];
end 

if sum(sum(dummies2))>0 && i==2
    x=[x,dummies2(p+1+h:T,:)];
end 

if smoothing == 1
if h>0
        y=y-shock_series(p+1:T-h,1:k)*prmtr_level_store(N*(h-1)+i,1:k)'; %subtract off level effect of MP shock to estimate change in parameter@ 
end 
end

x_hat=x; %just for scale of prior

end

p=p_max; %reset p back to original in case of long differences


%=================  PRIOR PARAMETERS ====================
    
%Normal Prior for LP intercept and coefficients
b0_m = zeros(size(x_hat,2),1);
b0_v = 10000000000*eye(size(x_hat,2));

    
%Minnesota like prior on level of IRFs, shrinking to zero at long horizons
if minnesota_lr==1   
b0_v(1:l,1:l) = (sige2(i)*(lambda_irf/(1+h))^2)*eye(l);
end    


%Smoothing priors on IRFs, with error-correction assumption for levels towards zero in long run
if smoothing == 1  
%b0_v(1:l,1:l) = sige2(i)*lambda_smooth^2;  %uncomment to place shrinkage prior on impact     
if h<peak && h>0   
b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+peak-h))^2)*eye(l);    
b0_m(1:l) = (1-gamma_lr^(peak-h))*prmtr_level_store(N*(h-1)+i,1:l)';
end
if h>=peak && h>0
b0_v(1:l,1:l) = (sige2(i)*(lambda_smooth/(base+h-peak))^2)*eye(l);    
b0_m(1:l) = (gamma_lr^(h-peak)-1)*prmtr_level_store(N*(h-1)+i,1:l)';
end 
end   
                                   

%Minnesota like shrinkage priors on lag coefficients -- set mean to one for own-lag
if minnesota_lags==1
j=1;
while j<=p
ii=1;
while ii<=N    
b0_v(l+2+(j-1)*N+ii,l+2+(j-1)*N+ii)=(sige2(i)/sige2(ii))*(lambda_lags/j)^2; %k+2 reflects IRF, constant, and trend first three elements of x                                                                           
if j==1 && ii==i
    b0_m(l+2+(j-1)*N+ii)=1; 
end 
ii=ii+1;
end      
j=j+1;
end
end

if ac == 1
if minnesota_controls == 1    
%Minnesota like prior on AC controls, with mean and variance shrinking to zero at longer horizons
if h>0
    j=0; jj=0;
    if h>control_truncation
        jj=h-control_truncation;
    end
    while jj<=h-1
            b0_m(b+1+j)=mean_decay^(h-1-jj);
            b0_v(b+1+j,b+1+j)=(lambda_controls/(h-jj))^2;
    j=j+1;
    jj=jj+1;
    end    
end
end
end

if ac == 2
if minnesota_controls == 1    
%Minnesota like prior on AC controls, with mean and variance shrinking to zero at longer horizons
if h>0
       b0_m(b+1:b+N)=zeros(N,1); b0_m(b+i)=1;
       b0_v(b+1:b+N,b+1:b+N)=(lambda_controls^2)*eye(N);  
end
end
end

%Informative prior on impact effect
if impact_prior==1
if h==0 & i==1
b0_m(1:l)=norm_sign(1:l);
b0_v(1:l,1:l)=(lambda_impact^2)*eye(l);
end
end


if iv == 1

%Normal Prior for first stage intercept and coefficients@
b0s1_m = zeros(size(z,2),1);
b0s1_v = 10000000000*eye(size(z,2));

if prior_iv==1
b0s1_m(1:k)=iv_prior_mean*norm_sign;
b0s1_v(1:k,1:k)=iv_prior_var*eye(k);
end

end

  
%Inverted Gamma parameters for sigma_e^2 
v0 = 0;
d0 = 0; 

v0_iv=v0;
d0_iv=d0;

if minnesota_variance == 1
%For informative, set v0>4, with higher tightening prior, and d0=(v0+2)*target_mode to target mode at sample variance from AR(4) times h+1
v0 = lambda_var;  %9 implies a standard deviation of about 1 if the mode is 1, note (v0,d0) correspond to 2*(alpha,beta) for standard representation of gamma distribution
v0_iv=v0;
if horizon_expand == 1
d0 = (v0+2)*sige2(i)*(h+1);
d0_iv=(v0_iv+2)*sige2(1)*(h+1);
else
d0 = (v0+2)*sige2(i);  
d0_iv=(v0_iv+2)*sige2(1);
end
end


if g==1
betagl = prmtr_point_store((i-1)*(h_max+1)+h+1,:)'; %use OLS or IV for initial value 
if iv ==1
betagl1 = prmtr1_point_store(1:l,:)'; %use 1st-stage OLS for initial value
betagl=[betagl;zeros(l,1)]; %add in parameters for coefficients on v's in 2nd-stage
end
if ac == 1
if h>0
    betagl=[betagl;zeros(min(h,control_truncation),1)];
end 
end
if ac == 2
if h>0
    betagl=[betagl;zeros(N,1)];
end
end
if leads==1 && h>0
    betagl=[betagl;zeros(k*min(h,p),1)];
end
if sum(sum(dummies1))>0 && i==1
    betagl=[betagl;zeros(size(dummies1,2),1)];
end 
if sum(sum(dummies2))>0 && i==2
    betagl=[betagl;zeros(size(dummies2,2),1)];
end  
else
if iv ==1
betagl1 = prmtr1_lag_store(1:l,:)';
betagl = prmtr_lag_store(N*h+i,1:n_xnd)';
else
betagl = prmtr_lag_store(N*h+i,1:n_xnd)';
end
if leads==1 && h>0
    betagl=[betagl;leads_lag_store(N*h*k+i,1:k*min(p,h))'];
end
if sum(sum(dummies1))>0 && i==1
    betagl=[betagl;dummies1_lag_store(h+1,:)'];
end
if sum(sum(dummies2))>0 && i==2
    betagl=[betagl;dummies2_lag_store(h+1,:)'];
end
end  

%=================START SAMPLING================

if iv ==1

%1st stage
if i==1 && h==0  %only need to conduct first stage regression once per draw of IRFs
v=[];
if l==1
sig2g1 = GEN_SIG(x(:,1),z,betagl1,v0_iv,d0_iv);
betag1 = GEN_BETA(x(:,1),z,sig2g1,b0s1_m,b0s1_v,i,h,l,n_x,norm_sign,sign_lag);
else
betag1=zeros(size(x,2),l);    
for j=1:l
    betagl1j=[betagl1(j,j);betagl1(l+1:end,j)];
    b0s1_mj=[b0s1_m(j);b0s1_m(l+1:end)];
    b0s1_vj=[b0s1_v(j,j);diag(b0s1_v(l+1:end,l+1:end))].*eye(length(b0s1_mj));

    xj=x(:,j);
    zj=[z(:,j),z(:,l+1:end)];
    zj=zj(xj~=0,:);
    xj=xj(xj~=0);

sig2g1 = GEN_SIG(xj,zj,betagl1j,v0_iv,d0_iv);
betag1j = GEN_BETA(xj,zj,sig2g1,b0s1_mj,b0s1_vj,i,h,l,n_x,norm_sign,sign_lag);
betag1(j,j) = betag1j(1);
betag1(l+1:end,j) = betag1j(2:end);
v=[v,x(:,j)-z*betag1(:,j)];

if j>1 && orthogonalize==1
vjb=(v(:,1:j-1)'*v(:,1:j-1)\v(:,1:j-1)'*v(:,j));
v(:,j)=v(:,j)-v(:,1:j-1)*vjb; %orthogonalize errors
end

end


end
end

%2nd stage
v=x(:,1:l)-z11*betag1;
x_hat(:,b-l+1:b)=v;  
if i==1 && h==0
betag= [norm_sign(1:l);zeros(size(x_hat,2)-size(instrumented_variables,2),1)];
else    
sig2g = GEN_SIG(y,x_hat,betagl,v0,d0);
betag = GEN_BETA(y,x_hat,sig2g,b0_m,b0_v,i,h,l,n_x,norm_sign,sign_lag);  
end 

else

sig2g = GEN_SIG(y,x,betagl,v0,d0);     
betag = GEN_BETA(y,x,sig2g,b0_m,b0_v,i,h,l,n_x,norm_sign,sign_lag);

end

%===============================================

%Record 1st order serial correlation of residuals
if iv == 1
x = x_hat; %use fitted x to save and look at correlation for 2nd stage errors
end
errors=y-x*betag;
errors_hg=[errors_hg,errors];
corr_out=serial_correlation_fcn(errors,1,0);
corr5_out=serial_correlation_fcn(errors,5,0);
corrg=[corrg;corr_out];
corr5g=[corr5g;corr5_out];

if ac == 1
errors_g=[errors_g,[zeros(p+h,1);(y-x*betag)]];
end

if ac == 2
if h>0    
errors_g(:,N*(h-1)+i)=[zeros(p+h,1);(y-x*betag)];
%uncomment below to use full projection error rather than errors
%errors_g(:,N*(h-1)+i)=[zeros(p+h,1);(y-x(:,1:b)*betag(1:b))]; 
end
end

if sum(sum(dummies2))>0 && i==2
    dummies2_lag_store(h+1,:)=betag(length(betag)-size(dummies2,2)+1:length(betag))';
    betag=betag(1:length(betag)-size(dummies2,2));
end

if sum(sum(dummies1))>0 && i==1
    dummies1_lag_store(h+1,:)=betag(length(betag)-size(dummies1,2)+1:length(betag))';
    betag=betag(1:length(betag)-size(dummies1,2));
end

if leads==1 && h>0
    leads_lag_store(N*h*k+i,1:k*min(p,h))=betag(length(betag)-k*min(p,h)+1:length(betag))';
    betag=betag(1:length(betag)-k*min(p,h));    
end

prmtr_lag_store(N*h+i,1:length(betag))=betag';
if iv == 1
prmtr1_lag_store(1:l,1:length(betag1))=betag1';
end

if smoothing == 1
irfdiffg=[irfdiffg;betag(1:k)];    
if h>0
    betag(1:l)=prmtr_level_store(N*(h-1)+i,1:l)'+betag(1:l); 
end
prmtr_level_store(N*h+i,1:length(betag))=betag';    
end 

irfg=[irfg;betag(1:l)];

i=i+1;
end

corr_i_out=(sum(sum(abs(corr(errors_hg))))-4)/(4^2-4);
corrhg=[corrhg;corr_i_out];
            
h=h+1;
end


if g>N0   %store draws after burnin
               
irfmm=[irfmm;irfg'];
if smoothing == 1
irfdiffmm=[irfdiffmm;irfdiffg'];
end
corrmm=[corrmm;corrg']; 
corr5mm=[corr5mm;corr5g'];
corrhmm=[corrhmm;corrhg'];

end


g=g+1;
end

%Calculate counterfactual paths for real GDP and inflation
endogenous_variable_2_counterfact=[];
endogenous_variable_3_counterfact=[];
for g=1:MM

%Store BLP results in same format as CLP
Bayes_irfs_g=[];

for h=0:h_max
Bayes_irfs_g = [Bayes_irfs_g, irfmm(g,l*N*h+1:l*N*h+l*N)'];
end

Bayes_irfs_g = Bayes_irfs_g';

%Note: do not adjust for normsign given these are un-normalized responses used for counterfactual calculations

for i=2
    y_counterfact=endogenous_variables(:,i);
    for t=h_max+1:T
        if l==1
        y_counterfact(t)=endogenous_variables(t,i)-Bayes_irfs_g(:,l*(i-1)+1)'*flipud(shock_series(t-h_max:t,1));
        else
        y_counterfact(t)=endogenous_variables(t,i)-Bayes_irfs_g(:,l*(i-1)+1)'*flipud(shock_series(t-h_max:t,1))-Bayes_irfs_g(:,l*(i-1)+2)'*flipud(shock_series(t-h_max:t,2));
        end
    end
    endogenous_variable_2_counterfact=[endogenous_variable_2_counterfact,y_counterfact];
end


for i=3
    y_counterfact=endogenous_variables(:,i);
    for t=h_max+1:T
        if l==1
        y_counterfact(t)=endogenous_variables(t,i)-Bayes_irfs_g(:,l*(i-1)+1)'*flipud(shock_series(t-h_max:t,1));
        else
        y_counterfact(t)=endogenous_variables(t,i)-Bayes_irfs_g(:,l*(i-1)+1)'*flipud(shock_series(t-h_max:t,1))-Bayes_irfs_g(:,l*(i-1)+2)'*flipud(shock_series(t-h_max:t,2));
        end
    end
    endogenous_variable_3_counterfact=[endogenous_variable_3_counterfact,y_counterfact];
end
end


posterior_moments_counterfact2=moments_fcn(endogenous_variable_2_counterfact',c_level);
posterior_moments_counterfact3=moments_fcn(endogenous_variable_3_counterfact',c_level);


counterfact2_mean=posterior_moments_counterfact2(:,1);
counterfact3_mean=posterior_moments_counterfact3(:,1);
counterfact2_lower=posterior_moments_counterfact2(:,4);
counterfact3_lower=posterior_moments_counterfact3(:,4);
counterfact2_upper=posterior_moments_counterfact2(:,5);
counterfact3_upper=posterior_moments_counterfact3(:,5);

if normalize == 1  %rescale irfs to be for unit shock in terms of impact on OCR
normalize_factor=[];
for i=1:l
    normalize_factor=[normalize_factor,mean(irfmm(:,i))]; %normalize with posterior mean to get identical results to OLS given diffuse/improper priors
    %normalize_factor=[normalize_factor,irfmm(:,i)]; %normalize taking first-stage uncertainty into account
end
for h=0:h_max
    for j=1:N
        for i=1:l
            irfmm(:,h*N*l+(j-1)*l+i)=norm_sign(i)*irfmm(:,h*N*l+(j-1)*l+i)./normalize_factor(:,i);
            if smoothing == 1
            irfdiffmm(:,h*N*k+(j-1)*l+i)=norm_sign(i)*irfdiffmm(:,h*N*l+(j-1)*l+i)./normalize_factor(:,i);
            end
        end
    end
end
    for i=1:l
    irfmm(:,i)=norm_sign(i).*ones(size(irfmm,1),1);  %uncomment when normalizing all other horizons and variables by posterior mean
    irfdiffmm(:,i)=norm_sign(i).*ones(size(irfmm,1),1); %uncomment when normalizing all other horizons and variables by posterior mean
    end
end

fnl_mat = irfmm;
if smoothing == 1
fnl_mat2 = irfdiffmm; 
end
       
fnl_mat=[fnl_mat,fnl_mat2];

posterior_moments=moments_fcn(fnl_mat,c_level);

%Store BLP results in same format as CLP
Bayes_irfs=[];
Bayes_ses=[];
Bayes_irfs_lower=[];
Bayes_irfs_upper=[];

for h=0:h_max
Bayes_irfs = [Bayes_irfs, posterior_moments(l*N*h+1:l*N*h+l*N,1)];
Bayes_ses = [Bayes_ses, posterior_moments(l*N*h+1:l*N*h+l*N,2)];
Bayes_irfs_lower = [Bayes_irfs_lower, posterior_moments(l*N*h+1:l*N*h+l*N,4)];
Bayes_irfs_upper = [Bayes_irfs_upper, posterior_moments(l*N*h+1:l*N*h+l*N,5)];
end

Bayes_irfs = Bayes_irfs';
Bayes_ses = Bayes_ses';
Bayes_irfs_lower = Bayes_irfs_lower';
Bayes_irfs_upper = Bayes_irfs_upper';

% Creates output file to store results
output = fopen('results.txt','w');

% Print table of results into 'results.txt'
fprintf(output, 'Confidence level for equal-tailed bands\n');
fprintf(output, '        %0.8f\n', c_level);

fprintf(output, 'mean, std dev, median, lower percentile, upper percentile\n');
fprintf(output, '        %0.8f        %0.8f        %0.8f        %0.8f         %0.8f\n', posterior_moments(:, 1:5));

fprintf(output, 'correlation between draws(1, 5, 10, 20 draws apart)\n');
fprintf(output, '        %0.8f        %0.8f        %0.8f        %0.8f\n', posterior_moments(:, 6:9));

fclose(output);


% Display confidence level
disp('confidence level for equal-tailed bands')
disp(c_level)

% Display parameter estimates
disp('mean ~ std dev ~ median ~ lower percentile ~ upper percentile')
disp(posterior_moments(:, 1:5))

% Display correlation of draws
disp('correlation between draws(1, 5, 10, 20 draws apart)')
disp(posterior_moments(:, 6:9))


%% PLOT RESULTS
xbars=[0 sign_lag+1];
rr=3;
cc=round(N/rr)+1;
zz=zeros(1,h_max+1);  
names1 = {'LP for Cash rate',
         'LP for GDP', 
         'LP for Inflation',
         'LP for TWI'};
names2 = {'BLP for Cash rate',
         'BLP for GDP', 
         'BLP for Inflation',
         'BLP for TWI'};

figure
for j=1:N   
axis tight
xlim([0 h_max]);
subplot(2,N,j)
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', irfs_lower(:,(j-1)*l+1); (h_max:-1:0)' flipud(irfs_upper(:,(j-1)*l+1))];
grpyat2=[(0:1:h_max)', irfs_lower(:,(j-1)*l+l); (h_max:-1:0)' flipud(irfs_upper(:,(j-1)*l+l))];
patch(grpyat2(:,1), grpyat2(:,2), [0.7 0.7 0.7],'edgecolor',[0.7 0.7 0.7]);
alpha(0.4)
patch(grpyat(:,1), grpyat(:,2), [0.4 0.4 0.4],'edgecolor',[0.4 0.4 0.4]);
alpha(0.3)
hold on
h2=plot(0:1:h_max, irfs(:,(j-1)*l+l), 'color',[0 0.5 0],'linestyle','-.','LineWidth', 1.4);
hold on
h3=plot(0:1:h_max, irfs(:,(j-1)*l+1), 'r-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
axis tight
title(names1{j},[num2str(100*c_level),'% bands']);

switch names1{j}
    case {char('LP for Cash rate'),char('LP for Inflation')}
      ylabel('ppts') 
   otherwise
      ylabel('log points')
end



end
for j=1:N   
axis tight
xlim([0 h_max]);
subplot(2,N,N+j)
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', Bayes_irfs_lower(:,(j-1)*l+1); (h_max:-1:0)' flipud(Bayes_irfs_upper(:,(j-1)*l+1))];
grpyat2=[(0:1:h_max)', Bayes_irfs_lower(:,(j-1)*l+l); (h_max:-1:0)' flipud(Bayes_irfs_upper(:,(j-1)*l+l))];
patch(grpyat2(:,1), grpyat2(:,2), [0.7 0.7 0.7],'edgecolor',[0.7 0.7 0.7]);
alpha(0.4)
patch(grpyat(:,1), grpyat(:,2), [0.4 0.4 0.4],'edgecolor',[0.4 0.4 0.4]);
alpha(0.3)
hold on
h3=plot(0:1:h_max, Bayes_irfs(:,(j-1)*l+l), 'color',[0 0.5 0],'linestyle','-.','LineWidth', 1.4);
hold on
h2=plot(0:1:h_max, Bayes_irfs(:,(j-1)*l+1), 'r-','LineWidth', 1.4);
% if j>1
% hold on
% patch([xbars(1) xbars(1), xbars(2) xbars(2)], [min(ylim) max(ylim) max(ylim) min(ylim)], [0.4 0.4 0.4],'edgecolor', [0.4 0.4 0.4],'FaceAlpha',0.3);
% end
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
axis tight
title(names2{j},[num2str(100*c_level),'% bands']);

switch names2{j}
   case {char('BLP for Cash rate'),char('BLP for Inflation')}
      ylabel('ppts') 
   otherwise
      ylabel('log points')
end

end

hold off


% x0=10;
% y0=10;
% width=650;
% height=600
% set(gcf,'position',[x0,y0,width,height])



posterior_moments_sc=moments_fcn(corrmm,0.95);

%Store results for serial correlation of residuals
sc_pe=[];
sc_sd=[];
sc_lower=[];
sc_upper=[];


for h=0:h_max
sc_pe = [sc_pe, posterior_moments_sc(N*h+1:N*h+N,1)];
sc_sd = [sc_pe, posterior_moments_sc(N*h+1:N*h+N,2)];
sc_lower = [sc_lower, posterior_moments_sc(N*h+1:N*h+N,4)];
sc_upper = [sc_upper, posterior_moments_sc(N*h+1:N*h+N,5)];
end

sc_pe = sc_pe';
sc_sd = sc_sd';
sc_lower = sc_lower';
sc_upper = sc_upper';

posterior_moments_sc=moments_fcn(corr5mm,0.95);

sc5_pe=[];
sc5_sd=[];
sc5_lower=[];
sc5_upper=[];

for h=0:h_max
sc5_pe = [sc5_pe, posterior_moments_sc(N*h+1:N*h+N,1)];
sc5_sd = [sc5_pe, posterior_moments_sc(N*h+1:N*h+N,2)];
sc5_lower = [sc5_lower, posterior_moments_sc(N*h+1:N*h+N,4)];
sc5_upper = [sc5_upper, posterior_moments_sc(N*h+1:N*h+N,5)];
end

sc5_pe = sc5_pe';
sc5_sd = sc5_sd';
sc5_lower = sc5_lower';
sc5_upper = sc5_upper';

%% PLOT RESULTS FOR SERIAL CORRELATION OF RESIDUALS
rr=3;
cc=round(N/rr)+1;
zz=zeros(1,h_max+1);  
names1 = {'Serial corr. (1st) for Cash rate',
         'Serial corr. (1st) for GDP', 
         'Serial corr. (1st) for Inflation',
         'Serial corr. (1st) for TWI'};
names2 = {'Serial corr. (5th) for Cash rate',
         'Serial corr. (5th) for GDP', 
         'Serial corr. (5th) for Inflation',
         'Serial corr. (5th) for TWI'};

figure
for j=1:N   
axis tight
ylim([-1 1]);
xlim([0 h_max]);
subplot(2,N,j)
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', sc_lower(:,j); (h_max:-1:0)' flipud(sc_upper(:,j))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
h2=plot(0:1:h_max, sc_pe(:,j), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
ylim([-1 1]);
xlim([0 h_max]);
axis tight
title(names1{j},'95% bands');

% switch names1{j}
%     case {char('Serial correlation of residuals for Cash rate'),char('Serial correlation of residuals for Inflation')}
%       ylabel('ppts') 
%    otherwise
%       ylabel('log points')
% end

end
for j=1:N   
axis tight
xlim([0 h_max]);
ylim([-1 1]);
subplot(2,N,N+j)
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', sc5_lower(:,j); (h_max:-1:0)' flipud(sc5_upper(:,j))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
h2=plot(0:1:h_max, sc5_pe(:,j), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
ylim([-1 1]);
%axis tight
title(names2{j},'95% bands');

% switch names2{j}
%    case {char('BLP for Cash rate'),char('BLP for Inflation')}
%       ylabel('ppts') 
%    otherwise
%       ylabel('log points')
% end



end



%% For multiple shocks, calculate difference in IRFs for same unit shock
%normalized to be positive and comparison to first shock
if l>1
irfmm_compare=[];    
for h=0:h_max
    for j=1:N
        for i=2:l
            irfmm_compare=[irfmm_compare,norm_sign(1)*irfmm(:,h*N*l+(j-1)*l+1)-norm_sign(i)*irfmm(:,h*N*l+(j-1)*k+i)];
        end
    end
end
     

posterior_moments_compare=moments_fcn(irfmm_compare,c_level);


%Store BLP results in same format as CLP
Bayes_irfs_compare=[];
Bayes_ses_compare=[];
Bayes_irfs_lower_compare=[];
Bayes_irfs_upper_compare=[];

for h=0:h_max
Bayes_irfs_compare = [Bayes_irfs_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,1)];
Bayes_ses_compare = [Bayes_ses_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,2)];
Bayes_irfs_lower_compare = [Bayes_irfs_lower_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,4)];
Bayes_irfs_upper_compare = [Bayes_irfs_upper_compare, posterior_moments_compare((l-1)*N*h+1:(l-1)*N*h+(l-1)*N,5)];
end

Bayes_irfs_compare = Bayes_irfs_compare';
Bayes_ses_compare = Bayes_ses_compare';
Bayes_irfs_lower_compare = Bayes_irfs_lower_compare';
Bayes_irfs_upper_compare = Bayes_irfs_upper_compare';


rr=3;
cc=round(N/rr)+1;
zz=zeros(1,h_max+1);  
names1 = {'BLP IRF compare for Cash rate',
         'BLP IRF compare for GDP', 
         'BLP IRF compare for Inflation',
         'BLP IRF compare for TWI'};

figure
for j=1:N   
axis tight
xlim([0 h_max]);
subplot(1,N,j)
h1=plot(0:1:h_max, zz, 'k-');
hold on
grpyat=[(0:1:h_max)', Bayes_irfs_lower_compare(:,(j-1)*(l-1)+1); (h_max:-1:0)' flipud(Bayes_irfs_upper_compare(:,(j-1)*(l-1)+1))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.7 0.7],'edgecolor',[0.6 0.6 0.6]);
alpha(0.3)
hold on
h2=plot(0:1:h_max, Bayes_irfs_compare(:,(j-1)*(l-1)+1), 'b-','LineWidth', 1.4);
hold on
h5=plot(0:1:h_max, zz, 'k-');
xlim([0 h_max]);
axis tight
title(names1{j},[num2str(100*c_level),'% bands']);

switch names1{j}
    case {char('BLP IRF compare for Cash rate'),char('BLP IRF compare for Inflation')}
      ylabel('ppts') 
   otherwise
      ylabel('log points')
end


end

hold off

end


if iv == 0 %do not report counterfactuals for IV case given normalized IRFs by construction

%Plot counterfactual paths

counterfact_mean=[counterfact2_mean,counterfact3_mean];
counterfact_lower=[counterfact2_lower,counterfact3_lower];
counterfact_upper=[counterfact2_upper,counterfact3_upper];


rr=3;
cc=round(N/rr)+1;
zz=zeros(1,h_max+1);  
names1 = {
         'Inflation Counterfactual'};

dates = data(h_max+1,1):0.25:data(end,1); dates=dates';

figure  
axis tight
%xlim([0 h_max]);
plot(dates,endogenous_variables(h_max+1:T,3),'-r','LineWidth',2);
hold on
grpyat=[dates, counterfact_lower(h_max+1:T,2); flipud(dates) flipud(counterfact_upper(h_max+1:T,2))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.9 1],'edgecolor', [0.7 0.9 1]);
alpha(0.3)
hold on
h2=plot(dates, counterfact_mean(h_max+1:T,2), 'b-','LineWidth', 1.4);
    set(gca,'FontSize',18)
    set(gca,'Layer', 'top')
    title('Variance Decompositions','FontSize',18)   
    xlim([dates(1) dates(end)])
    axis tight
title('Inflation Counterfactual',['Actual (red) versus Counterfactual without MP Shocks (blue, with 68% bands)'],'FontSize',18)     

switch names1{1}
    case {char('Inflation Counterfactual')}
      ylabel('ppts') 
   otherwise
      ylabel('log points')
end

hold off


rr=3;
cc=round(N/rr)+1;
zz=zeros(1,T-h_max); 
names1 = {'Monetary Policy Shocks'};

dates = data(h_max+1,1):0.25:data(end,1); dates=dates';

figure    
axis tight
plot(dates,shock_series(h_max+1:T,:),'-k','LineWidth',2);
hold on
h2=plot(dates, zz, 'k-');
hold on
    set(gca,'FontSize',18)
    set(gca,'Layer', 'top')
    ylabel('ppts')
    xlim([dates(1) dates(end)])
    axis tight
title('Monetary Policy Shocks','FontSize',18)  


hold off


rr=3;
cc=round(N/rr)+1;
zz=zeros(1,T-h_max); 
names1 = {'Monetary Policy Shocks', 
         'Inflation Counterfactual'};


dates = data(h_max+1,1):0.25:data(end,1); dates=dates';

figure    
axis tight
subplot(2,1,1);
plot(dates,shock_series(h_max+1:T,:),'-k','LineWidth',2);
hold on
h1=plot(dates, zz, 'k-');
hold on
    set(gca,'FontSize',18)
    set(gca,'Layer', 'top')
    ylabel('ppts')
    xlim([dates(1) dates(end)])
    axis tight
title('Monetary Policy Shocks','FontSize',18)  


subplot(2,1,2);
plot(dates,endogenous_variables(h_max+1:T,3),'-r','LineWidth',2);
hold on
grpyat=[dates, counterfact_lower(h_max+1:T,2); flipud(dates) flipud(counterfact_upper(h_max+1:T,2))];
patch(grpyat(:,1), grpyat(:,2), [0.7 0.9 1],'edgecolor', [0.7 0.9 1]);
alpha(0.3)
hold on
h3=plot(dates, counterfact_mean(h_max+1:T,2), 'b-','LineWidth', 1.4);
    set(gca,'FontSize',18)
    set(gca,'Layer', 'top')  
    xlim([dates(1) dates(end)])
    ylabel('ppts') 
    axis tight
title('Inflation Counterfactual',['Actual (red) versus Counterfactual without MP Shocks (blue, with 68% bands)'],'FontSize',18)     
   
hold off

end