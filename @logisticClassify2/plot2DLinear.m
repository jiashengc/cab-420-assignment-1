function plot2DLinear(obj, X, Y)
    % plot2DLinear(obj, X,Y)
    %   plot a linear classifier (data and decision boundary) when features X are 2-dim
    %   wts are 1x3,  wts(1)+wts(2)*X(1)+wts(3)*X(2)
    %
    [n,d] = size(X);
    if (d~=2) error('Sorry -- plot2DLogistic only works on 2D data...'); end;
    
    wts = getWeights(obj);
    classes = unique(Y);
    Xline = linspace(min(X(:,1)),max(X(:,1)), 2);
    
    figure;
    hold on;
    scatter(X((Y==classes(1)),1),X((Y==classes(1)),2),50,'r*');
    scatter(X((Y==classes(2)),1),X((Y==classes(2)),2),50,'b*');

    plot(Xline, -obj.wts(1)/obj.wts(3) - obj.wts(2)/obj.wts(3) .* Xline, 'r-');
    legend('Positive Class', 'negative class');
    hold off;

end