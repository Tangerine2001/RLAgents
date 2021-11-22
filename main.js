function swapLinks(clicked) {
   const links = document.getElementById("links");
   links.innerHTML = generateDivElements(clicked);
}

function generateDivElements(clicked) {
    //Outer container to use to set innerHTML
    const linkContainer = document.createElement("div")
    linkContainer.classList.add("link-container")

    //Links container
    const sidebarLinks = document.createElement("div");
    sidebarLinks.classList.add("sidebar-links");

    // The three links here
    const act = document.createElement("div");
    act.classList.add("active");
    const inactive1 = document.createElement("div");
    inactive1.classList.add("inactive");
    const inactive2 = document.createElement("div");
    inactive2.classList.add("inactive");

    // Create the fancy arrows.
    const arrow1 = document.createElement("i");
    arrow1.classList.add("fas");
    arrow1.classList.add("fa-caret-right");
    const arrow2 = document.createElement("i");
    arrow2.classList.add("fas");
    arrow2.classList.add("fa-caret-right")
    const arrow3 = document.createElement("i");
    arrow3.classList.add("fas");
    arrow3.classList.add("fa-caret-right");

    // Create the actual text contents
    const home = document.createElement("div");
    home.classList.add("bp-text");
    home.textContent = "Home";
    home.setAttribute('onclick', "swapLinks('Home')");
    home.style.marginLeft = '16px';
    const about = document.createElement("div");
    about.classList.add("bp-text");
    about.textContent = "About";
    about.setAttribute('onclick', "swapLinks('About')");
    about.style.marginLeft = '16px';
    const cc = document.createElement("div");
    cc.classList.add("bp-text");
    cc.textContent = "ClassicControl";
    cc.setAttribute('onclick', "swapLinks('ClassicControl')");
    cc.style.marginLeft = '16px';

    // Add arrows to each of the bullet points
    act.appendChild(arrow1);
    inactive1.appendChild(arrow2);
    inactive2.appendChild(arrow3);

    // Assign text to arrows and links based on clicked input
    if (clicked === "Home") {
        act.appendChild(home);
        inactive1.appendChild(about);
        inactive2.appendChild(cc);

        sidebarLinks.appendChild(act);
        sidebarLinks.appendChild(inactive1);
        sidebarLinks.appendChild(inactive2);
    } else if (clicked ==="About") {
        inactive1.appendChild(home);
        act.appendChild(about);
        inactive2.appendChild(cc);

        sidebarLinks.appendChild(inactive1);
        sidebarLinks.appendChild(act);
        sidebarLinks.appendChild(inactive2);
    } else {
        inactive1.appendChild(home);
        inactive2.appendChild(about);
        act.appendChild(cc);

        sidebarLinks.appendChild(inactive1);
        sidebarLinks.appendChild(inactive2);
        sidebarLinks.appendChild(act);
    }

    linkContainer.appendChild(sidebarLinks)
    return linkContainer.innerHTML;
}
