%*****************************************************************************80
%
%% kursiv_etdrk4() solves the Kuramoto-Sivashinsky PDE using the ETD RK4 method.
%
%  Discussion:
%
%    The system being solved is:
%
%    ut = - u * ux - uxx - uxxx, periodic BC on 0 < x < 32 pi
%
%  Modified:
%
%    12 April 2020
%
%  Author:
%
%    Lloyd Trefethen
%
%  Reference:
%
%    Stephen Cox, Paul Matthews,
%    Exponential time differencing for stiff systems,
%    Journal of Computational Physics,
%    Volume 176, pages 430-455, 2002.
%
%    Aly-Khan Kassam, Lloyd Trefethen,
%    Fourth-order time-stepping for stiff ODE's,
%    SIAM Journal on Scientific Computing,
%    Volume 26, Number 4, pages 1214-1233, 2005.
%
%    Lloyd Trefethen,
%    Spectral methods in MATLAB,
%    SIAM, 2000,
%    LC: QA377.T65
%    ISBN: 978-0-898714-65-4
%
%  Input:
%
%    integer NX: the number of nodes.
%
%  Output:
%
%    real XX(NX+1): the spatial grid
%
%    real TT(NT): the time values.
%
%    real UU(NX,NT):

nx = 4096;
batch_size = 5000;
length = 64.0 * pi;
u = -1+2 * rand(nx, batch_size, 1); % cos (x/2); % cos ( x / 16.0 ) .* ( 1.0 + sin ( x / 16.0 ) );
tstart = 100;
tend = 121;
save_step = 1;
rng(123);
lam = 0;
dt = 0.25;

c8_i = sqrt ( -1.0 );
x = length * ( 1 : nx )' / nx;
force = lam * cos(2 * pi * x);
v = fft ( u );
%
%  Precompute various ETD RK4 scalar quantities.
%
k = [ 0:nx/2-1, 0.0, -nx/2+1:-1 ]' * 2 * pi / length;
L = k.^2 - k.^4;
E = exp ( dt * L );
E2 = exp ( dt * L / 2.0 );
M = 32;
%
%  Roots of unity.
%
r = exp ( c8_i * pi * ( (1:M) - 0.5 ) / M );
%
%  Note row vector + column vector operation!
%
LR = dt * L(:,ones(M,1)) + r(ones(nx,1),:);

Q  = dt * real ( mean ( ...
( exp ( LR / 2.0 ) - 1.0 ) ./ LR, 2 ) );
f1 = dt * real ( mean ( ...
 ( - 4.0 - LR + exp ( LR ) .* ( 4.0 - 3.0 * LR + LR.^2 )) ./ LR.^3, 2 ) );
f2 = dt * real ( mean ( ...
( 2.0 + LR + exp ( LR ) .* ( - 2.0 + LR ) ) ./ LR.^3, 2 ) );
f3 = dt * real ( mean ( ...
( - 4.0 - 3.0 * LR - LR.^2 + exp ( LR ) .* ( 4.0 - LR ) ) ./ LR.^3, 2 ) );
%
%  Time stepping.
%
uu = u;
tt = 0.0;
nmax = round (tend / dt );
g = - 0.5 * c8_i * k;

for i = 1 : nmax

    t = i * dt;
    
    Nv = g .* fft ( real ( ifft ( v ) ) .^2 );
    a = E2 .* v + Q .* Nv;
    Na = g .* fft ( real ( ifft ( a ) ) .^2 );
    b = E2 .* v + Q .* Na;
    Nb = g .* fft ( real ( ifft ( b ) ) .^2 );
    c = E2 .* a + Q .* ( 2.0 * Nb - Nv );
    Nc = g .* fft ( real ( ifft ( c ) ) .^2 );
    
    v = E .* v + Nv .* f1 + 2.0 * ( Na + Nb ) .* f2 + Nc .* f3;
    
    u = real ( ifft ( v ) ) + force;
    if ( t>=tstart && mod ( i, save_step ) == 0 )
      uu = cat(4,uu,u);
      tt = [ tt, t ];
    end

end

%uu = permute(uu, [4,2,3,1]);
%save(['./1dks_', num2str(floor((tend - tstart) / (dt * save_step))), 'x', num2str(batch_size), 'x1x', num2str(nx), '_dt', num2str(dt * save_step), '_t[', num2str(tstart), '_', num2str(tend), ']', '.mat'], 'uu');


%% reload data from file
uu=load('1dks_20x1200x1x4096_dt1_t[100_600].mat','uu').uu;
uu=permute(uu,[4,2,3,1]); %(4096,1200,1,20)

%%
% XX = 0:length/(nx):length-length/(nx);
% YY = tstart:dt*save_step:tend-dt*save_step;
for batch=1:2
    % Go back in real space omega in real space for plotting
    u=squeeze(uu(:,batch,1,1:(tend-tstart)/(dt*save_step)));
    %contourf(w,50); colorbar; shading flat;colormap('jet'); 
    pcolor(u); shading flat; colorbar; 
    %pcolor(XX,YY,w); shading flat; axis equal tight; colorbar;
    % path=['./batch', num2str(batch), '/fig', sprintf('%04d',mode(k,100)), '.jpg'];
    % if ~exist(['./batch', num2str(batch)],'dir')
    % mkdir(['./batch', num2str(batch)]);
    % end
    % saveas(gcf,path)
    drawnow;
end
