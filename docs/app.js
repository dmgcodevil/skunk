const filterInput = document.querySelector("#nav-filter");
const navLinks = Array.from(document.querySelectorAll("#doc-nav a"));
const sections = navLinks
    .map((link) => {
        const id = link.getAttribute("href")?.slice(1);
        const section = id ? document.getElementById(id) : null;
        return section ? { link, section, label: link.textContent.toLowerCase() } : null;
    })
    .filter(Boolean);

if (filterInput) {
    filterInput.addEventListener("input", () => {
        const query = filterInput.value.trim().toLowerCase();
        for (const item of sections) {
            const match = item.label.includes(query);
            item.link.classList.toggle("is-hidden", !match);
        }
    });
}

if (sections.length > 0) {
    const observer = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];

            if (!visible) {
                return;
            }

            for (const item of sections) {
                item.link.classList.toggle("active", item.section === visible.target);
            }
        },
        {
            rootMargin: "-20% 0px -55% 0px",
            threshold: [0.15, 0.4, 0.7],
        }
    );

    for (const item of sections) {
        observer.observe(item.section);
    }
}
