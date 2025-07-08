// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', () => {
    // 导航栏交互
    const navbar = document.querySelector('.navbar');
    
    // 滚动监听
    window.addEventListener('scroll', () => {
        navbar.style.background = window.scrollY > 100 
            ? 'rgba(255, 255, 255, 0.98)' 
            : 'rgba(255, 255, 255, 0.95)';
    });

    // 汉堡菜单切换
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');
    
    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('show');
        hamburger.classList.toggle('active');
    });
    
    // 平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // 元素入场动画
    const animateElements = document.querySelectorAll('.timeline-item, .skill-card, .project-card');
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = 1;
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, { threshold: 0.1 });

    animateElements.forEach(el => {
        el.style.opacity = 0;
        el.style.transform = 'translateY(20px)';
        observer.observe(el);
    });
});

// 其他功能函数
function handleContactSubmit(e) {
    e.preventDefault();
    // 表单提交处理逻辑
}