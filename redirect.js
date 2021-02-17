var base_url = "https://www.adriangb.com/scikeras/";

var current = window.location.href;
var rest = current.replace(base_url, "");

var target_version = null;

$.getJSON(base_url + "versions.json", function (versions) {
    if (versions.hasOwnProperty("stable")) {
        target_version = versions.stable;
        return;
    }
    if (versions.hasOwnProperty("latest")) {
        target_version = versions.latest;
        return;
    }
});

function maybeRedirect() {
    if (((target_version !== null) && !(rest.startsWith("refs")))) {
        // redirect
        window.location.replace(base_url.concat(target_version, "/", rest));
    }
}

window.addEventListener('load', function () {
    // Multiple timeouts, fails if it runs too soon
    setTimeout(maybeRedirect, 25);
    setTimeout(maybeRedirect, 50);
    setTimeout(maybeRedirect, 100);
    setTimeout(maybeRedirect, 200);
    setTimeout(maybeRedirect, 500);
}, false);
