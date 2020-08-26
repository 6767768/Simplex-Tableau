function [flg, r] = Revisedgetr(n,s,B,T,t)
%
% find the index to kick out
%
% On input:
% B: current basis index
% T: current Revised Tableau
% t: current pivot column
% s: s-th  column is to JOIN the index.
% n: number of unknowns.
%

% On output:
%
% r: B(r)-th column is to LEAVE the index.
%    r < 0 indicates unbounded LP.
%

flg = 0;
x   = zeros(n,1);
x(B)= T(1:end-1,1);
if (max(t)<n*eps) % LP has no minimized solution
    r = -1;
    return
end
mask        = find(t>0);% return the index of t>0, sift for those qualified entries.
[lambda, r] = min(x(B(mask))./t(mask));
r           = mask(r); %find index r in the tableau instead of index in the vector mask.
if (lambda < 1e-14) %degenerate LP, for degenerate LP, the pivot element is 0.
    flg  = 1;
end
return
