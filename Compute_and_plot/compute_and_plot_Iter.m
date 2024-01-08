% plot routine used for error plots of full range
% (plots median value, filled area between min and max,
% thicker area between 0.25- and 0.75-quantile)

% Input:
% -'x_arrays': cell array with entries of size 'num_iters' (something) x
%   num_repeats, each field corresponds to a method. Contains reference
%   values such as iterations, runtime, number of multiplications.
%   Also allowed: array of size 'num_iters', then
%   we generate a cell array with 'num_methods' many copies
%   Also allowed: array of size 'num_iters' x num_methods, then
%   we generate a cell array with 'num_methods' many copies
% -'arrs': cell array with entries of size 'num_iters' x num_repeats,
%   each field corresponds to a method. Contains error quantities such as
%   errors ||x-x_hat||
%   Also allowed: 3d array, where each slice corresponds to a method, then
%   we generate a cell array with 'num_methods' many copies
% - 'num_methods' = length(arrs) == length(x_arrays)
% - 'line_colors', ... are cell arrays which specifies a color, ... for each
%   method

function [x_arrays, quantiles] = compute_and_plot_Iter(x_arrays, arrs, ...
    num_methods, line_colors, display_names, ...
    minmax_colors, quant_colors, display_legend, max_val_in_plot)

if ~iscell(x_arrays)
    new_x_arrays = cell(1, num_methods);
    if size(x_arrays, 2) == 1 && size(x_arrays, 3) == 1 % first expectional case in description
        for i = 1:num_methods
            new_x_arrays{i} = x_arrays;
        end
    elseif size(x_arrays, 2) == num_methods && size(x_arrays, 3) == 1 % second expectional case in description
        for i = 1:num_methods
            new_x_arrays{i} = x_arrays(:,i);
        end
    else
        error('This format of x_arrays is not supported!')
    end
    x_arrays = new_x_arrays;
end

if ~iscell(arrs)
    assert(size(arrs,3) == num_methods)
    new_arrs = cell(1, num_methods);
    for i = 1:num_methods
        new_arrs{i} = arrs(:,:,i);
    end
    arrs = new_arrs;
end

% cut away rows with NaN entries, zero entries and entries > 'max_val_in_plot'
for i = 1:length(arrs)
    if any(any(arrs{i} < eps) | any(arrs{i} > max_val_in_plot))
        row_has_no_nan_entries = all(~isnan(arrs{i}'));
        arrs{i} = arrs{i}(row_has_no_nan_entries, :);
        x_arrays{i} = x_arrays{i}(row_has_no_nan_entries);
        row_has_low_entries = all(arrs{i}' < max_val_in_plot);
        if isempty(find(row_has_low_entries))
            error('arrs{%d} has no values < max_val_in_plot!', i)
        end
        arrs{i} = arrs{i}(row_has_low_entries, :);
        x_arrays{i} = x_arrays{i}(row_has_low_entries);
    end
end

% compute quantiles
[mins, maxs, medians, quant25s, quant75s] = compute_minmax_median_quantiles(arrs);
quantiles = {mins, maxs, medians, quant25s, quant75s};

% plot
figure
hold on

plot_array = zeros(1, num_methods);

for i = 1:num_methods

    h = fill([x_arrays{i}'  fliplr(x_arrays{i}')], [maxs{i}' fliplr(mins{i}')], minmax_colors{i},'EdgeColor', 'none');
    set(h,'facealpha', .5)

    h = fill([x_arrays{i}'  fliplr(x_arrays{i}')], [quant75s{i}'  fliplr(quant25s{i}')], quant_colors{i},'EdgeColor', 'none');
    set(h,'facealpha', .5)

    if i==1
        plot_array(i) = plot( x_arrays{i}, medians{i}, line_colors{i}, 'LineWidth', 1,...
            'LineStyle', '-','Marker', '^', 'DisplayName', display_names{i});
    else if i==2
            plot_array(i) = plot( x_arrays{i}, medians{i}, line_colors{i}, 'LineWidth', 1,...
                'LineStyle', '-','Marker', 'd', 'DisplayName', display_names{i});
    else if i==3
            plot_array(i) = plot( x_arrays{i}, medians{i}, line_colors{i}, 'LineWidth', 1,...
                'LineStyle', '-','Marker', 'o', 'DisplayName', display_names{i});

    else
        plot_array(i) = plot( x_arrays{i}, medians{i}, line_colors{i}, 'LineWidth', 1,...
            'LineStyle', '-','Marker', 'p', 'DisplayName', display_names{i});
    end
    end
    end

end

%if display_legend
%    legend(plot_array, display_names, 'location', 'northwest', 'FontSize', 7, 'location', 'best');
%end

if display_legend
    legend(plot_array, display_names,'location', 'best');
end

% on y axis: replace t by 10^t for interesting t values

%       min_tick = inf;
%       max_tick = -inf;
%       for i = 1:num_methods
%           min_tick = min(min_tick, min(mins{i}));
%           max_tick = max(max_tick, max(maxs{i}));
%       end
%       min_tick = floor(log10(min_tick));
%       max_tick = ceil(log10(max_tick));
%       new_YTick = [0, min_tick, round(0.5*(min_tick+max_tick)), max_tick];
%       new_YTick = sort(unique(new_YTick));
%       YTickLabel = cell(1,length(new_YTick));
%       for ii = 1:length(new_YTick)
%           if new_YTick(ii) == 0
%               YTickLabel{ii} = '0';
%           else
%               YTickLabel{ii} = num2str(new_YTick(ii), '10^{%d}');
%           end
%       end
%
%     ylim([min_tick max_tick])
%     set(gca, 'YTick', new_YTick, 'YTickLabel', YTickLabel);
%     drawnow
%axis square



    function [mins, maxs, medians, quant25s, quant75s] = compute_minmax_median_quantiles(arrs)
        [mins, maxs, medians, quant25s, quant75s] = deal(cell(1,length(arrs)));
        for method = 1:length(arrs)  % loop over all methods
            arr = arrs{method};
            if size(arr, 2) == 1   % if only one repeat, nothing to compute
                [mins{method}, maxs{method}, medians{method}, quant25s{method} ,quant75s{method}]...
                    = deal(arr);
            else
                mins{method} = min(arr, [], 2);
                maxs{method} = max(arr, [], 2);
                medians{method} = median(arr,2);
                quant25s{method} = quantile(arr,0.25,2);
                quant75s{method} = quantile(arr,0.75,2);
            end
        end
    end

end