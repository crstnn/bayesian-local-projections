function var_draw = GEN_SIG(y,x,betagl,v0,d0)
% James Morley
% Updated 31 July 2026
% Draws residual variance from conditional posterior for linear regression
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%betagl   Previous draw of regression coefficients
%v0       prior hyperparameter
%d0       prior hyperparameter
%
%
%% OUTPUTS
%var_draw        Draw of residual variance from inverse gamma
    
e_mat=y-x*betagl;

v1 = v0 + length(e_mat);
d1 = d0 + e_mat'*e_mat;
c = rndc(v1);
t2 = c/d1;
var_draw=1/t2;
