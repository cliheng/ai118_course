// 证书数据
const awards = [
    "🏆 金猪奖年度技术先锋（2023）",
    "🛡️ 五灵侠守护勋章（连续5年获得者）",
    "💻 全球超能开发者大赛冠军（2022）",
    "⚛️ 量子计算黑客松特等奖（2021）"
];

// 初始化证书
function initAwards() {
    const container = document.getElementById('awards-container');
    awards.forEach(award => {
        const div = document.createElement('div');
        div.textContent = award;
        div.style.cursor = 'pointer';
        div.onclick = () => alert(`获奖详情：${award}`);
        container.appendChild(div);
    });
}

// 主题切换功能
function toggleTheme() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('theme', 
        document.body.classList.contains('dark-mode') ? 'dark' : 'light');
}

// 初始化主题
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'light';
    if (savedTheme === 'dark') document.body.classList.add('dark-mode');
}

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initAwards();
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
});