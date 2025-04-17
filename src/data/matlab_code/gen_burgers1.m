% number of realizations to generate
N = 100000;

% parameters for the Gaussian random field
gamma = 2.5;
tau = 7;
sigma = 7^(2);
T = 1;

% viscosity
visc = 1/100;

% grid size
s = 128 * 1;
steps = 200 * 1;


input = zeros(N, s);
if steps == 1
    sol = zeros(N, s);
else
    sol = zeros(N, steps, s, 'single');
end

tspan = linspace(0,T,steps+1);
x = linspace(0,1,s+1);
for j=1:N
    u0 = GRF1(s/2, 0, gamma, tau, sigma, "periodic");
    u = burgers1(u0, tspan, s, visc);
    
    ic = u0(x);
    input(j,:) = ic(1:end-1);
    
    if steps == 1
        sol(j,:) = u.values;
    else
        sol(j,1,:) = input(j, :);
        for k=2:(steps+1)
            sol(j,k,:) = u{k}.values;
        end
    end
    
    disp(j);
end
disp("Done computing")
save('Burgers_v_0.01_N_100000_T_1_size_128.mat', 'sol', '-v7.3');
disp("Done saving")

