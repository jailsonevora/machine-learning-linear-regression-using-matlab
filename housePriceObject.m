classdef housePriceObject
    %HOUSEPRICE Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        data
        X 
        y
        theta
        alpha
        num_iters
        m
        iterations
        mu
        sigma
    end    
    
    methods(Access = public, Static = true)
        
%         function obj = housePrice(data,X,y,theta,alpha,num_iters,m,iterations,mu,sigma)
%             %UNTITLED Construct an instance of this class
%             %   Detailed explanation goes here
%             obj.data = data;
%             obj.X = X;
%             obj.y = y;
%             obj.theta = theta;
%             obj.alpha = alpha;
%             obj.num_iters = num_iters;
%             obj.m = m;
%             obj.iterations = iterations;
%             obj.mu = mu;
%             obj.sigma = sigma;            
%         end
        
        function plotData(X, y)
            figure; % open a new figure window
            plot(X, y, 'rx', 'MarkerSize', 10); % Plot the data
            ylabel('Profit in $10,000s'); % Set the y-axis label
            xlabel('Population of City in 10,000s'); % Set the x-axis label
        end
        
        function J = computeCost(X, y, theta)  
        % Vetorize
            J = 1/(2*this.m) * sum((X*theta - y).^2);
        end
        
        function J = computeCostMulti(X, y, theta)
            J = 1/(2*this.m) * sum((X * theta - y).^2);        
        end
        
        function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
        J_history = zeros(num_iters, 1);

            for iter = 1:num_iters                
                thetaTemp0 = theta(1) - (alpha/length(y)) * sum((X*theta - y));
                thetaTemp1 = theta(2) - (alpha/length(y)) * sum((X*theta - y).*X(:,2));

                theta(1)= thetaTemp0;
                theta(2)= thetaTemp1;

                % Save the cost J in every iteration    
                J_history(iter) = computeCost(X, y, theta);
            end
        end
        
        function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
        J_history = zeros(num_iters, 1);
            for iter = 1:num_iters
                theta = theta - (alpha/length(y)) * ((X * theta - y)' * X)';
                % Save the cost J in every iteration    
                J_history(iter) = computeCostMulti(X, y, theta);
            end
        end
        
        function [X_norm, mu, sigma] = featureNormalize(X)        
            mu = mean(X);
            sigma = std(X);

            X = (X - ones(size(X,1),1)*mu) ./ (ones(size(X,1),1)*sigma);
            X_norm = X;       
        end
        
        function [theta] = normalEqn(X, y)        
            theta =  pinv(X' * X) * (X'*y);
        end      
    end
end



    

