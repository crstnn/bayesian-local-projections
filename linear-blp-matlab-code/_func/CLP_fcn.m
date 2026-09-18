function [irf_store,se_store,prmtr_point_store,b1] = CLP_fcn(endogenous_variables,controls,block_exogenous_controls,shock_series,l,orthogonalize,dummies1,dummies2,norm_sign,iv,h_max,p,neweywest,ld,leads)
% James Morley
% Updated 4 August 2026
% Calculates classical local projection estimates of IRFs
%
%% INPUTS
%
%endogenous_variables     Endogenous variables to project
%controls                 Other variables to include as controls
%block_exogenous_controls Other variables to include as controls, including contemporaneous
%shock_series             Shock or instrument
%l                        Number of shocks or instrumented series
%dummies1                 Dummy variables for ZLB
%dummies2                 Dummy variables for Covid
%norm_sign                Normalization of sign of shock(s)
%iv                       iv=1 means conduct IV instead of OLS
%h_max                    maximum horizon to consider for IRFs
%p                        # of lags of controls to include
%nw                       nw=1 means calculate Newey West standard errors,otherwise standard OLS or IV
%ld                       ld=1 means long difference specification
%leads                    leads=1 means include some min(h,p) leads, no longer SUR unless residuals uncorrelated
%
%% OUTPUTS
%irf_store                Estimated impulse responses from horizon 0 to hmax
%se_store                 Standard errors for estimated IRFs
%pmrtr_point_store        vector of point estimates for all regression parameters

iv_approach=3; %1=IV, 2=2SLS, 3=2SLSCF

T = length(endogenous_variables);
N = size(endogenous_variables,2); %number of endogenous variables
k=size(shock_series,2); %number of shocks series
p_max=p;

irf_store=[];
se_store=[];
prmtr_point_store=[];

b1=[];

i=1;        %indicator for LHS variable  
while i <= N 
       
bh=[];
seh=[];

h=0;  %horizon for LP
while h <= h_max
   
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
    if ld ==1
    x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
    end
end
    j=1;
    while j <= p_max
        x=[x,endogenous_variables(p+1-j:T-h-j,:)];  %lags of output, inflation, and OCR
        if ld == 1
        x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,:)-endogenous_variables(p-j:T-h-1-j,:);    
        end
        if controls ~= 0
            x=[x,controls(p+1-j:T-h-j,:)];
        if ld == 1
        x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);    
        end
        end        
        if block_exogenous_controls ~= 0
            x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
        if ld == 1
        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);    
        end
        end 
        j=j+1;
    end   

z=[shock_series(p+1:T-h,:),x(:,l+1:size(x,2))];
z11=z;

if sum(sum(dummies1))>0
    z11=[z11,dummies1(p+1+h:T,:)];
end

if leads==1    
    j=1;
    while j <= min(p_max,h)
        x=[x,shock_series(p+1+j:T-h+j,:)]; %leads of MP shocks
    j=j+1;
    end 
end

if sum(sum(dummies1))>0 && i==1
    x=[x,dummies1(p+1+h:T,:)];
    z=[z,dummies1(p+1+h:T,:)];
end

if sum(sum(dummies2))>0 && i==2
    x=[x,dummies2(p+1+h:T,:)];
    z=[z11,dummies2(p+1+h:T,:)];
end

if iv_approach==1
[b,u,~,se,x_hat] = iv_fcn(y,x,z);
elseif iv_approach==2
[b,b1,u,~,se,x_hat] = tsls_fcn(y,x,z,i,h,b1,z11,l);
else
[b,b1,u,~,se,x_hat] = tslscf_fcn(y,x,z,i,h,b1,z11,l,orthogonalize);
end
x = x_hat; %use fitted x for Newey West correction with IV estimates

else

x=[shock_series(p+1:T-h,:),ones(T-p-h,1),(1:1:T-p-h)'];   %mp shock, constant, time trend@
if block_exogenous_controls ~= 0
    x=[x,block_exogenous_controls(p+1:T-h,:)];
    if ld ==1
    x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1:T-h,:)-block_exogenous_controls(p:T-h-1,:);
    end
end
    j=1;
    while j <= p_max
        x=[x,endogenous_variables(p+1-j:T-h-j,1:N)];  %lags of OCR, output, and inflation
        if ld == 1
        x(:,size(x,2)-N+1:size(x,2))=endogenous_variables(p+1-j:T-h-j,1:N)-endogenous_variables(p-j:T-h-1-j,1:N);    
        end
        if controls ~= 0
            x=[x,controls(p+1-j:T-h-j,:)];
        if ld == 1
        x(:,size(x,2)-size(controls,2)+1:size(x,2))=controls(p+1-j:T-h-j,:)-controls(p-j:T-h-1-j,:);    
        end
        end        
        if block_exogenous_controls ~= 0
            x=[x,block_exogenous_controls(p+1-j:T-h-j,:)];
        if ld == 1
        x(:,size(x,2)-size(block_exogenous_controls,2)+1:size(x,2))=block_exogenous_controls(p+1-j:T-h-j,:)-block_exogenous_controls(p-j:T-h-1-j,:);    
        end
        end 
        if leads==1
            x=[x,shock_series(p+1-j:T-h-j,:)];  
        end
        j=j+1;
    end       

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

[b,u,~,se] = ols_fcn(y,x);

end

if neweywest == 1
    se = neweywest_fcn(x,b,u,h);
end

if iv==1 && iv_approach==3
    b = b(1:length(b)-l);
    se = se(1:length(se)-l);
end    

if sum(sum(dummies2))>0 && i==2
    b = b(1:length(b)-size(dummies2,2));
    se = se(1:length(se)-size(dummies2,2));
end

if sum(sum(dummies1))>0 && i==1
    b = b(1:length(b)-size(dummies1,2));
    se = se(1:length(se)-size(dummies1,2));
end

if leads==1 && h>0
    b = b(1:length(b)-k*min(p_max,h));
    se = se(1:length(se)-k*min(p_max,h));
end
        
bh=[bh;b(1:l)'];
seh=[seh;se(1:l)'];   

prmtr_point_store = [prmtr_point_store;b'];
     
h=h+1;
end

irf_store=[irf_store,bh];
se_store=[se_store,seh];

i=i+1;
end 

