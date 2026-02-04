/**
 * Alchemist2026 Web - 图表工具函数
 */

// 图表颜色配置
const CHART_COLORS = {
    up: '#26a69a',      // 上涨 - 绿色
    down: '#ef5350',    // 下跌 - 红色
    line: '#2196f3',    // 默认线条 - 蓝色
    grid: '#e0e0e0',    // 网格线
};

// 多资产对比颜色
const COMPARISON_COLORS = [
    '#2196f3',  // 蓝色
    '#ff9800',  // 橙色
    '#4caf50',  // 绿色
    '#9c27b0',  // 紫色
    '#f44336',  // 红色
    '#00bcd4',  // 青色
    '#795548',  // 棕色
    '#607d8b',  // 灰蓝色
];

/**
 * 格式化数字
 * @param {number} num - 数字
 * @param {number} decimals - 小数位数
 * @returns {string}
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return num.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/**
 * 格式化日期
 * @param {string} dateStr - ISO 日期字符串
 * @returns {string}
 */
function formatDate(dateStr) {
    if (!dateStr) return '-';
    return dateStr.split('T')[0];
}

/**
 * 格式化成交量
 * @param {number} volume - 成交量
 * @returns {string}
 */
function formatVolume(volume) {
    if (volume >= 1e9) {
        return (volume / 1e9).toFixed(2) + 'B';
    } else if (volume >= 1e6) {
        return (volume / 1e6).toFixed(2) + 'M';
    } else if (volume >= 1e3) {
        return (volume / 1e3).toFixed(2) + 'K';
    }
    return volume.toFixed(0);
}

/**
 * 创建 K 线图配置
 * @param {Array} data - OHLCV 数据数组
 * @param {string} symbol - 资产代码
 * @returns {Object} Plotly trace 配置
 */
function createCandlestickTrace(data, symbol) {
    return {
        type: 'candlestick',
        x: data.map(d => d.timestamp),
        open: data.map(d => d.open),
        high: data.map(d => d.high),
        low: data.map(d => d.low),
        close: data.map(d => d.close),
        increasing: {
            line: { color: CHART_COLORS.up },
            fillcolor: CHART_COLORS.up
        },
        decreasing: {
            line: { color: CHART_COLORS.down },
            fillcolor: CHART_COLORS.down
        },
        name: symbol
    };
}

/**
 * 创建成交量图配置
 * @param {Array} data - OHLCV 数据数组
 * @returns {Object} Plotly trace 配置
 */
function createVolumeTrace(data) {
    const colors = data.map(d =>
        d.close >= d.open ? CHART_COLORS.up : CHART_COLORS.down
    );

    return {
        type: 'bar',
        x: data.map(d => d.timestamp),
        y: data.map(d => d.volume),
        marker: { color: colors },
        name: '成交量',
        hovertemplate: '%{x}<br>成交量: %{y:,.0f}<extra></extra>'
    };
}

/**
 * 创建折线图配置
 * @param {Array} data - OHLCV 数据数组
 * @param {string} symbol - 资产代码
 * @param {number} colorIndex - 颜色索引
 * @returns {Object} Plotly trace 配置
 */
function createLineTrace(data, symbol, colorIndex = 0) {
    const color = COMPARISON_COLORS[colorIndex % COMPARISON_COLORS.length];

    return {
        type: 'scatter',
        mode: 'lines',
        x: data.map(d => d.timestamp),
        y: data.map(d => d.close),
        name: symbol,
        line: { color: color, width: 2 },
        hovertemplate: `${symbol}<br>%{x}<br>价格: $%{y:.2f}<extra></extra>`
    };
}

/**
 * 创建归一化收益率折线图配置
 * @param {Array} data - OHLCV 数据数组
 * @param {string} symbol - 资产代码
 * @param {number} colorIndex - 颜色索引
 * @returns {Object} Plotly trace 配置
 */
function createReturnsTrace(data, symbol, colorIndex = 0) {
    const color = COMPARISON_COLORS[colorIndex % COMPARISON_COLORS.length];
    const basePrice = data[0].close;

    return {
        type: 'scatter',
        mode: 'lines',
        x: data.map(d => d.timestamp),
        y: data.map(d => ((d.close - basePrice) / basePrice) * 100),
        name: symbol,
        line: { color: color, width: 2 },
        hovertemplate: `${symbol}<br>%{x}<br>收益率: %{y:.2f}%<extra></extra>`
    };
}

/**
 * 默认图表布局配置
 * @returns {Object} Plotly layout 配置
 */
function getDefaultLayout() {
    return {
        margin: { t: 20, b: 40, l: 60, r: 20 },
        xaxis: {
            type: 'date',
            gridcolor: CHART_COLORS.grid,
            showgrid: true
        },
        yaxis: {
            gridcolor: CHART_COLORS.grid,
            showgrid: true
        },
        showlegend: true,
        legend: {
            orientation: 'h',
            y: -0.15
        },
        hovermode: 'x unified'
    };
}

/**
 * 计算简单统计数据
 * @param {Array} data - OHLCV 数据数组
 * @returns {Object} 统计数据
 */
function calculateStats(data) {
    if (!data || data.length === 0) {
        return null;
    }

    const closes = data.map(d => d.close);
    const highs = data.map(d => d.high);
    const lows = data.map(d => d.low);

    const startPrice = closes[0];
    const endPrice = closes[closes.length - 1];
    const returns = ((endPrice - startPrice) / startPrice) * 100;

    return {
        startPrice: startPrice,
        endPrice: endPrice,
        returns: returns,
        high: Math.max(...highs),
        low: Math.min(...lows),
        count: data.length
    };
}
