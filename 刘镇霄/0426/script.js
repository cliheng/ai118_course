// 导航栏滚动效果
window.addEventListener('scroll', function () {
    const header = document.getElementById('header');
    if (window.scrollY > 100) {
        header.classList.add('scrolled');
    } else {
        header.classList.remove('scrolled');
    }
});

// 移动端菜单切换
const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');

hamburger.addEventListener('click', function () {
    navLinks.classList.toggle('active');
});

// 关闭移动菜单当点击链接
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', () => {
        navLinks.classList.remove('active');
    });
});

// 表单提交
const contactForm = document.querySelector('.contact-form form');
if (contactForm) {
    contactForm.addEventListener('submit', function (e) {
        e.preventDefault();            // 获取表单数据
        const name = document.getElementById('name').value;
        const email = document.getElementById('email').value;
        const subject = document.getElementById('subject').value;
        const message = document.getElementById('message').value;

        // 这里可以添加表单验证逻辑

        // 模拟表单提交
        console.log('表单已提交:', {name, email, subject, message});

        // 显示成功消息
        alert('感谢您的留言！我会尽快回复您。');

        // 重置表单
        contactForm.reset();
    });
}

// 动画效果 - 当元素进入视口时触发
const animateOnScroll = function () {
    const elements = document.querySelectorAll('.skill-item, .project-card, .education-item, .timeline-item');

    elements.forEach(element => {
        const elementPosition = element.getBoundingClientRect().top;
        const screenPosition = window.innerHeight / 1.3;

        if (elementPosition < screenPosition) {
            element.style.opacity = '1';
            element.style.transform = 'translateY(0)';
        }
    });
};

// 初始化元素样式
window.addEventListener('DOMContentLoaded', function () {
    document.querySelectorAll('.skill-item, .project-card, .education-item, .timeline-item').forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(20px)';
        element.style.transition = 'all 0.5s ease';
    });

    // 立即检查一次
    animateOnScroll();
});

// 滚动时触发动画
window.addEventListener('scroll', animateOnScroll);

// 平滑滚动
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();

        const targetId = this.getAttribute('href');
        const targetElement = document.querySelector(targetId);

        if (targetElement) {
            window.scrollTo({
                top: targetElement.offsetTop - 80,
                behavior: 'smooth'
            });
        }
    });
});
