function x = rndc(a)
% James Morley
% Updated 31 July 2026
% Draws from Chisquared (and, more generally, Gamma) distribution
%
%% INPUTS
%
%a        degrees of freedom for Chisquared
%
%
%% OUTPUTS
%x        Draw from Chisquared
    
a = a/2;
if a > 1
w = rg2(a);
elseif a < 1
a = a + 1;
u = rand;
w = rg2(a)*u^(1/a);
elseif a == 1
w = -ln(rand);
end
x = w * 2;