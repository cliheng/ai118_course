// 平滑滚动到页面的指定部分
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();

        document.querySelector(this.getAttribute('href')).scrollIntoView({
            behavior: 'smooth'
        });
    });
});

// 简单的动画效果（例如，页面加载时的淡入效果）
window.addEventListener('load', () => {
    const sections = document.querySelectorAll('section');
    sections.forEach(section => {
        section.style.opacity = 0;
        section.style.transition = 'opacity 1s ease-in-out';

        // 延迟动画，避免同时出现
        setTimeout(() => {
            section.style.opacity = 1;
        }, 1000);
    });
});

// 导航栏固定效果
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    const hero = document.querySelector('.hero');
    const heroHeight = hero.offsetHeight;

    if (window.scrollY > heroHeight) {
        navbar.classList.add('fixed');
    } else {
        navbar.classList.remove('fixed');
    }
});

// 为技能添加鼠标悬停放大效果
document.querySelectorAll('.skill-item').forEach(skill => {
    skill.addEventListener('mouseover', () => {
        skill.style.transform = 'scale(1.1)';
        skill.style.transition = 'transform 0.3s ease';
    });

    skill.addEventListener('mouseout', () => {
        skill.style.transform = 'scale(1)';
    });
});

// 为项目展示添加鼠标悬停阴影效果
document.querySelectorAll('.project-item').forEach(project => {
    project.addEventListener('mouseover', () => {
        project.style.boxShadow = '0 5px 15px rgba(0, 0, 0, 0.3)';
        project.style.transition = 'box-shadow 0.3s ease';
    });

    project.addEventListener('mouseout', () => {
        project.style.boxShadow = 'none';
    });
});