function [b,b1,u,sigu2,se,x_hat] = tslscf_fcn(y,x,z,i,h,b1,z11,l,orthogonalize)
% James Morley
% Updated 4 August 2026
% Calculates 2SLS estimates using control function, residuals, and standard errors
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%z        Instruments
%
%
%% OUTPUTS
%b        IV point estimates
%u        residuals based on IV estimates
%sigu2    residual variance
%se       IV SEs
%x_hat    fitted x


if i==1 && h==0

bs1 = (z'*z)\(z'*x);

if l>1
x1=x(:,1);
z1=[z(:,1),z(:,3:end)];
x2=x(:,2);
z2=z(:,2:end);

bs11 = (z1(x1~=0,:)'*z1(x1~=0,:))\(z1(x1~=0,:)'*x1(x1~=0));
bs11=[bs11(1);0;bs11(2:end)];
bs12 = (z2(x2~=0,:)'*z2(x2~=0,:))\(z2(x2~=0,:)'*x2(x2~=0));
bs12=[0;bs12(1:end)];

    bs1(:,1:2)=[bs11,bs12];
    
end

b1=bs1;
z11=z;

else

bs1=b1;

end

if l==1
    v=x(:,1)-z11*bs1(:,1);
else
    v1=x(:,1)-z11*bs1(:,1);
    v2=x(:,2)-z11*bs1(:,2);
    v2b=(v1'*v1)\(v1'*v2);
    v2_orth=v2;
    if orthogonalize==1
    v2_orth=v2-v1*v2b;
    end
    v=[v1,v2_orth];
end
    
x_hat=[x,v];
b = (x_hat'*x_hat)\(x_hat'*y);   
u = y-x_hat*b;
sigu2 = (u'*u)/(length(u)-length(b));   
se = sqrt(diag(sigu2*inv(x_hat'*x_hat)));
