
function [data,info]=LPRevised(A,b,c)

format short g;

data=struct;
info=struct;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PhaseI
[m,n]=size(A);

if rank(A)<m %If matrix A is not full row rank, return degenerate failure.
    info.run='Failure';
    info.msg='In Phase I,failure due to degeneracy.(Matrix A is not full row rank.)';
    return
end

B=(n+1:n+m)';
Ahat=[A eye(m)];
chat=[zeros(n,1);ones(m,1)];
xhat=[zeros(n,1);b];
T = Ahat(:,B)\[b eye(m)];
y = T(:,2:end)'*chat(B);
T = [T;[chat'*xhat,y']];
n=n+m;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PhaseI tableau
% disp('Starting Phase I Simplex Iteration... ');
% disp('Initial Basis is');
% disp(B');
% obj = chat'*xhat;
% disp(['Initial Objective = ', num2str(obj)]);
% disp('Displaying Initial solution x, c-A^T*y and their componentwise product');
% disp([x c-A'*y x.*(c-A'*y)]);
simplex = 1;
ITER = 0;
%pause(2);

while (simplex == 1)
    %
    % determine the next s and r values.
    %
    y        = T(end,2:end)';
    [zmin,s] = min(chat-Ahat'*y);
    %
    % check for convergence.
    %
    if (abs(zmin) < 1e-14)
        
        %disp('Simplex Method has converged');
        simplex = 0;
        %         disp('Displaying Optimal Basis');
        %         disp(B');
        xhat   = zeros(n,1);
        xhat(B) = T(1:end-1,1);
        obj  = chat'*xhat;
        
        if (any(isnan(xhat))==1||any(isnan(y))==1||any(isinf(xhat))==1||any(isinf(y))==1)
            info.run='Failure';
            info.msg='In Phase I, failure due to arithmetic exceptions.';
            return
        end
        
        %         disp(['Optimal Objective = ', num2str(obj),' after ', num2str(ITER), ' iterations']);
        %         disp('Displaying Optimal solution x, c-A^T*y and their componentwise product');
        %         disp([x c-A'*y x.*(c-A'*y)]);
        info.run='Success';
        info.PhaseI.loop=ITER;
        data.PhaseI.obj=obj;
        data.PhaseI.x=xhat;
        continue;
    end
    
    t        = T(1:end-1,2:end)*Ahat(:,s);
    [flg,r] = Revisedgetr(n,s,B,T,t);
    if (flg == 1)
        %disp('LP is degenerate');
        info.run='Failure';
        info.msg='In Phase I,failure due to degeneracy.(b is a linear combination of fewer than m columns of A)';
        return
    end
    %When in Phase I, negative infinite won't happen.
    %     if (r < 1)
    %         %disp('LP has no lower bound');
    %         info.run=["Success"];
    %         info.case=2;
    %         continue;
    %end
    xhat   = zeros(n,1);
    xhat(B)= T(1:end-1,1);
    ITER = ITER + 1;
    %f = ['Iteration ', num2str(ITER), ' Obj ', num2str(c'*x), '. Smallest component in c-A^T*y: ', ...
    %num2str(zmin), ' at s =', num2str(s), '. Component r = ', num2str(r), ' out of basis'];
    %   disp(f);
    obj1 = chat'*xhat;
    %
    % update the revised simplex tableau.
    %
    [T,B1,flg]=RevisedSimplexTableau(B,r,s,t,zmin,T);
    if (flg == 1)
        %disp('LP is degenerate');
        info.run='Failure';
        info.msg='In Phase I,failure due to degeneracy.(b is a linear combination of fewer than m columns of A)';
        return
    end
    B   = B1;
    obj = obj1;
    %   disp('Current Basis is');
    %   disp(B');
    %   pause(1);
end
%disp('Phase I ends');

%Case 3
if (abs(obj)>1e-14)
    info.run='Success';
    info.case=3;
    data.PhaseI.obj=obj;
    info.PhaseI.loop=ITER;
    data.PhaseI.x=xhat;
    return
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Phase II
[m,n]=size(A);

x    = zeros(n,1);
x(B) = T(1:end-1,1);
T = A(:,B)\[b eye(m)];
y = T(:,2:end)'*c(B);
T = [T;[c'*x,y']];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Phase II tableau
% disp('Starting Phase II Simplex Iteration... ');
%
% disp('Initial Basis is');
% disp(B');
% obj = c'*x;
% disp(['Initial Objective = ', num2str(obj)]);
%disp([x c-A'*y x.*(c-A'*y)]);
simplex = 1;
ITER = 0;
%pause(2);

while (simplex == 1)
    %
    % determine the next s and r values.
    %
    y        = T(end,2:end)';
    [zmin,s] = min(c-A'*y);
    %
    % check for convergence.
    %
    if (abs(zmin) < 1e-14)
        simplex = 0;
        %         disp('Displaying Optimal Basis');
        %         disp(B');
        x   = zeros(n,1);
        x(B) = T(1:end-1,1);
        obj  = c'*x;
        
        if (any(isnan(x))==1||any(isnan(y))==1||any(isinf(x))==1||any(isinf(y))==1)
            info.run='Failure';
            info.msg='In Phase II, failure due to arithmetic exceptions.';
            return
        end
        %         disp(['Optimal Objective = ', num2str(obj),' after ', num2str(ITER), ' iterations']);
        %         disp('Displaying Optimal solution x, c-A^T*y and their componentwise product');
        %         disp([x c-A'*y x.*(c-A'*y)]);
        info.run='Success';
        info.case=1;
        data.PhaseII.Primalobj=obj;
        data.PhaseII.Dualobj =obj;
        data.PhaseII.x=x;
        data.PhaseII.y=y;
        data.PhaseII.z=c-A'*y;
        info.PhaseII.loop=ITER;
        continue;
    end
    
    t        = T(1:end-1,2:end)*A(:,s);
    [flg,r] = Revisedgetr(n,s,B,T,t);
    if (flg == 1)
        %disp('LP is degenerate');
        %simplex = 0;
        info.run='Failure';
        info.msg='In Phase II,failure due to degeneracy.(b is a linear combination of fewer than m columns of A)';
        return
    end
    if (r < 1)
        %disp('LP has no lower bound');
        %simplex = 0;
        %return t vector
        tstar=zeros(n,1);
        tstar(B)=t;
        tstar(s)=-1;
        
        info.run='Success';
        info.case=2;
        data.PhaseII.x=x;
        data.PhaseII.t=tstar;
        info.PhaseII.loop=ITER;
        return
    end
    x   = zeros(n,1);
    x(B)= T(1:end-1,1);
    ITER = ITER + 1;
    %f = ['Iteration ', num2str(ITER), ' Obj ', num2str(c'*x), '. Smallest component in c-A^T*y: ', ...
    %num2str(zmin), ' at s =', num2str(s), '. Component r = ', num2str(r), ' out of basis'];
    %   disp(f);
    obj1 = c'*x;
    %
    % update the revised simplex tableau.
    %
    [T,B1,flg]=RevisedSimplexTableau(B,r,s,t,zmin,T);
    if (flg == 1)
        %disp('LP is degenerate');
        %simplex = 0;
        info.run='Failure';
        info.msg='In Phase II,failure due to degeneracy.(b is a linear combination of fewer than m columns of A)';
        return
    end
    B   = B1;
    obj = obj1;
    %   disp('Current Basis is');
    %   disp(B');
    %   pause(1);
end

%disp('Phase II ends');

clear B1 f obj1 t zmin


