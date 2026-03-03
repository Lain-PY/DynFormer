clear all;

% Initialize parallel pool if not already running
if isempty(gcp('nocreate'))
    parpool('local');
end

rng(123);
nu=1e-5;  % viscosity
NX=64;     % resolution in x
NY=64;     % resolution in y
dt=1e-3;    % time step
save_time_start=10;
TF=30;  % final time
TSCREEN=1/dt; % sreen update interval time (NOTE: plotting is usually slow)
method='crank_nicholson';       % 'Forward_Euler' or 'crank_nicholson' or 'rk4'
initial_condition='random';   % 'vortices' or 'random'
save_t_step=1000;
batch_size=1200;
I=sqrt(-1);
dx=2*pi/(NX);
dy=2*pi/(NY);

XX = 0:2*pi/(NX):2*pi-dx;
YY = 0:2*pi/(NY):2*pi-dy;

switch lower(method)
   case {'forward_euler'}
      disp('Forward Euler method')
      disp('WARNING: only conditionally stable !!! Lower the time step if unstable...')
   case {'crank_nicholson'}
      disp('Crank-Nicholson Method')
      disp('Unconditionally stable up to CFL Condition')
   case {'rk4'}
      disp('the classical RK4 Method')
      disp('Lower the time step if unstable...')
    otherwise
      disp('Unknown method!!!');
      return
end

% Define initial vorticity distribution
switch lower(initial_condition)
   case {'vortices'}
      [i,j]=meshgrid(1:NX,1:NY);
      w=exp(-((i*dx-pi).^2+(j*dy-pi+pi/4).^2)/(0.2))+exp(-((i*dx-pi).^2+(j*dy-pi-pi/4).^2)/(0.2))-0.5*exp(-((i*dx-pi-pi/4).^2+(j*dy-pi-pi/4).^2)/(0.4));
   case {'random'}
      w=-1+2 * rand(NX, NY, batch_size, 1);
   otherwise
      disp('Unknown initial conditions !!!');
      return
end

kx=I*ones(1,NY)'*(mod((1:NX)-ceil(NX/2+1),NX)-floor(NX/2)) .* 2*pi; % matrix of wavenumbers in x direction 
ky=I*(mod((1:NY)'-ceil(NY/2+1),NY)-floor(NY/2))*ones(1,NX) .* 2*pi; % matrix of wavenumbers in y direction 

dealias=kx<2/3*NX&ky<2/3*NY; % Cutting of frequencies using the 2/3 rule

ksquare_viscous=kx.^2+ky.^2;        % Laplacian in Fourier space
ksquare_poisson=ksquare_viscous;    
ksquare_poisson(1,1)=1;             % fixed Laplacian in Fourier space for Poisson's equation

% Prepare data structures for parallel processing
num_save_steps = floor((TF - save_time_start)/(dt*save_t_step));

% Pre-allocate arrays for results
ws_result = zeros(NX, NY, num_save_steps, batch_size);
u_result = zeros(NX, NY, num_save_steps, batch_size);
v_result = zeros(NX, NY, num_save_steps, batch_size);
kin_seq = zeros(floor(TF/dt/TSCREEN));

% Convert w to cell array for parallel processing
w_cell = cell(batch_size, 1);
for b = 1:batch_size
    if ndims(w) == 3
        w_cell{b} = w(:,:,b);
    else
        w_cell{b} = w;
    end
end

% Process each batch in parallel
disp('Starting parallel processing...');
parfor b = 1:batch_size
    % Initialize local variables for this batch
    w_local = w_cell{b};
    w_hat_local = fft2(w_local);
    
    % Arrays to store results for this batch
    ws_local = zeros(NX, NY, num_save_steps);
    u_local = zeros(NX, NY, num_save_steps);
    v_local = zeros(NX, NY, num_save_steps);
    
    % Initialize variables needed in the loop
    w_hat_new_local = w_hat_local; % Initialize to avoid the warning
    
    % Time stepping loop for this batch
    k_local = 0;
    t_local = 0;
    save_t_local = 0;
    
    while t_local < TF+dt
        % Compute the stream function and get the velocity and gradient of vorticity
        psi_hat_local = -w_hat_local./ksquare_poisson;  % Solve Poisson's Equation
        u_temp = real(ifft2(ky.*psi_hat_local));      % Compute y derivative of stream function ==> u
        v_temp = real(ifft2(-kx.*psi_hat_local));     % Compute -x derivative of stream function ==> v
        w_x_local = real(ifft2(kx.*w_hat_local));     % Compute x derivative of vorticity
        w_y_local = real(ifft2(ky.*w_hat_local));     % Compute y derivative of vorticity
        conv_local = u_temp.*w_x_local + v_temp.*w_y_local;  % evaluate the convective derivative (u,v).grad(w)   
        conv_hat_local = fft2(conv_local);            % go back to Fourier space
        conv_hat_local = dealias.*conv_hat_local;     % Perform spherical dealiasing 2/3 rule
        
        % Save data at specified intervals
        if t_local >= save_time_start && mod(k_local, save_t_step) == 0
            ws_local(:,:,save_t_local+1) = real(ifft2(w_hat_local));
            u_local(:,:,save_t_local+1) = u_temp;
            v_local(:,:,save_t_local+1) = v_temp;
            save_t_local = save_t_local + 1;
        end

        % Compute Solution at the next step
        switch lower(method)
           case {'forward_euler'}
              w_hat_new_local = w_hat_local + dt*(nu*ksquare_viscous.*w_hat_local-conv_hat_local);
           case {'crank_nicholson'}
              w_hat_new_local = ((1/dt + 0.5*nu*ksquare_viscous)./(1/dt - 0.5*nu*ksquare_viscous)).*w_hat_local - (1./(1/dt - 0.5*nu*ksquare_viscous)).*conv_hat_local;
           case {'rk4'}
              k1 = (nu*ksquare_viscous.*w_hat_local-conv_hat_local);

              tempw_hat = w_hat_local+k1*dt/2;
              w_x_temp = real(ifft2(kx.*tempw_hat));  
              w_y_temp = real(ifft2(ky.*tempw_hat));      
              conv_temp = u_temp.*w_x_temp + v_temp.*w_y_temp;         
              conv_hat_temp = fft2(conv_temp);              
              conv_hat_temp = dealias.*conv_hat_temp;   
              k2 = (nu*ksquare_viscous.*tempw_hat-conv_hat_temp);

              tempw_hat = w_hat_local+k2*dt/2;
              w_x_temp = real(ifft2(kx.*tempw_hat));  
              w_y_temp = real(ifft2(ky.*tempw_hat));      
              conv_temp = u_temp.*w_x_temp + v_temp.*w_y_temp;         
              conv_hat_temp = fft2(conv_temp);              
              conv_hat_temp = dealias.*conv_hat_temp;
              k3 = (nu*ksquare_viscous.*tempw_hat-conv_hat_temp);

              tempw_hat = w_hat_local+k3*dt;
              w_x_temp = real(ifft2(kx.*tempw_hat));  
              w_y_temp = real(ifft2(ky.*tempw_hat));      
              conv_temp = u_temp.*w_x_temp + v_temp.*w_y_temp;         
              conv_hat_temp = fft2(conv_temp);              
              conv_hat_temp = dealias.*conv_hat_temp;
              k4 = (nu*ksquare_viscous.*tempw_hat-conv_hat_temp);

              w_hat_new_local = w_hat_local + dt/6*(k1+2*k2+2*k3+k4);
        end

        % Update for next time step
        w_hat_local = w_hat_new_local;
        t_local = t_local + dt;
        k_local = k_local + 1;
    end
    
    % Store results from this batch in separate arrays
    ws_result(:,:,:,b) = ws_local;
    u_result(:,:,:,b) = u_local;
    v_result(:,:,:,b) = v_local;
end

% Reshape the results to match the expected output format
ws = zeros(NX, NY, num_save_steps, batch_size, 1);
uv = zeros(NX, NY, num_save_steps, batch_size, 2);

% Copy data from parallel results to output arrays
for b = 1:batch_size
    ws(:,:,:,b,1) = ws_result(:,:,:,b);
    uv(:,:,:,b,1) = u_result(:,:,:,b);
    uv(:,:,:,b,2) = v_result(:,:,:,b);
end

% For visualization, use the first batch
radius = sqrt(abs(kx).^2+abs(ky).^2);
w_hat = fft2(w_cell{1});
t = 0;
k = 0;

% Visualization loop (only for the first batch)
while t < TF+dt
    % Compute the stream function and get the velocity and gradient of vorticity
    psi_hat = -w_hat./ksquare_poisson;  % Solve Poisson's Equation
    u = real(ifft2(ky.*psi_hat));      % Compute y derivative of stream function ==> u
    v = real(ifft2(-kx.*psi_hat));     % Compute -x derivative of stream function ==> v
    w_x = real(ifft2(kx.*w_hat));      % Compute x derivative of vorticity
    w_y = real(ifft2(ky.*w_hat));      % Compute y derivative of vorticity
    conv = u.*w_x + v.*w_y;            % evaluate the convective derivative (u,v).grad(w)   
    conv_hat = fft2(conv);             % go back to Fourier space
    conv_hat = dealias.*conv_hat;      % Perform spherical dealiasing 2/3 rule
    
    % Compute Solution at the next step
    switch lower(method)
       case {'forward_euler'}
          w_hat_new = w_hat + dt*(nu*ksquare_viscous.*w_hat-conv_hat);
       case {'crank_nicholson'}
          w_hat_new = ((1/dt + 0.5*nu*ksquare_viscous)./(1/dt - 0.5*nu*ksquare_viscous)).*w_hat - (1./(1/dt - 0.5*nu*ksquare_viscous)).*conv_hat;
       case {'rk4'}
          k1 = (nu*ksquare_viscous.*w_hat-conv_hat);

          tempw_hat = w_hat+k1*dt/2;
          w_x = real(ifft2(kx.*tempw_hat));  
          w_y = real(ifft2(ky.*tempw_hat));      
          conv = u.*w_x + v.*w_y;         
          conv_hat = fft2(conv);              
          conv_hat = dealias.*conv_hat;   
          k2 = (nu*ksquare_viscous.*tempw_hat-conv_hat);

          tempw_hat = w_hat+k2*dt/2;
          w_x = real(ifft2(kx.*tempw_hat));  
          w_y = real(ifft2(ky.*tempw_hat));      
          conv = u.*w_x + v.*w_y;         
          conv_hat = fft2(conv);              
          conv_hat = dealias.*conv_hat;
          k3 = (nu*ksquare_viscous.*tempw_hat-conv_hat);

          tempw_hat = w_hat+k3*dt;
          w_x = real(ifft2(kx.*tempw_hat));  
          w_y = real(ifft2(ky.*tempw_hat));      
          conv = u.*w_x + v.*w_y;         
          conv_hat = fft2(conv);              
          conv_hat = dealias.*conv_hat;
          k4 = (nu*ksquare_viscous.*tempw_hat-conv_hat);

          w_hat_new = w_hat + dt/6*(k1+2*k2+2*k3+k4);
    end
    
    % Plotting the vorticity field
    if mod(k,TSCREEN) == 0
        % Go back in real space omega in real space for plotting
        w = real(ifft2(w_hat_new));
        %contourf(w,50); colorbar; shading flat;colormap('jet'); 
        pcolor(XX,YY,w); shading flat; axis equal tight; colorbar; 
        %pcolor(XX,YY,w); shading flat; axis equal tight; colorbar;
        title(num2str(t));
        drawnow
        kinetic = u.^2+v.^2;
        kin_seq(floor(t / dt / TSCREEN)+1) = mean(kinetic,'all');
    end
    
    w_hat = w_hat_new;
    t = t + dt;
    k = k + 1;
end

ws = permute(ws, [4,3,5,1,2]);
uv = permute(uv, [4,3,5,1,2]);

save(['./2dturb_', num2str(batch_size), 'x', num2str(floor((TF - save_time_start) / (dt * save_t_step))), 'x1x', num2str(NX), 'x', num2str(NY), '_dt', num2str(dt * save_t_step), '_t[', num2str(save_time_start), '_', num2str(TF), ']_nu', num2str(nu), '.mat'], 'ws', 'uv');
