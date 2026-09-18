function beta_draw = GEN_BETA(y,x,sig2g,b0_m,b0_v,i,h,k,n_x,norm_sign,sign_lag)
% James Morley
% Updated 31 July 2026
% Draws regression coefficients from conditional posterior
%
%% INPUTS
%
%y        LHS variable
%x        RHS variables
%sig2g    Previous draw of residual variance
%v0       prior mean
%d0       prior variance
%
%
%% OUTPUTS
%beta_draw     Draw of regression coefficients from multivariate normal

  ystar=y./sqrt(sig2g);
  xstar=x./sqrt(sig2g);

    % b1_v=(b0_v\eye(length(b0_m)) + xstar'*xstar)\eye(length(b0_m));
    % b1_m=b1_v*(b0_v\b0_m + xstar'*ystar);

    b1_v=inv(inv(b0_v) + xstar'*xstar);
    b1_m=b1_v*(inv(b0_v)*b0_m + xstar'*ystar);

    accept=0;

    while accept==0

    c=chol(b1_v);
    beta_draw=b1_m + c'*randn(size(x,2),1);

accept=1;
if i>1 && h<=sign_lag
    for j=1:k
        if n_x>0    % for varying fiscal regimes with/without sign-asymmetry
            if (i==2 || i==3) && (norm_sign(j)*beta_draw(j)>0 || norm_sign(j)*beta_draw(n_x+j)>0)  %sign restriction that GDP and inflation fall in first sign_lag quarters
            accept=0;
            end
            
            if (i==4) && (norm_sign(j)*beta_draw(j)<0 || norm_sign(j)*beta_draw(n_x+j)<0)  %sign restrictions on the first sign_lag quarters to remove exchange rate puzzle by construction
            accept=0;
            end
       
        end

        if n_x==0 % for the linear and sign-asymmetry cases    
        
            if (i==2 || i==3) && norm_sign(j)*beta_draw(j)>0  %sign restriction that GDP and inflation falls
            accept=0;
            end

            if (i==4) && (norm_sign(j)*beta_draw(j)<0) %sign restriction to remove exchange rate puzzle
            accept=0;
            end
        end
    end    
end

    end