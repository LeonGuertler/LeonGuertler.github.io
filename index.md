---
layout: page
title: "👋 Welcome"
---

A minimal AI notebook, rendered entirely in **Ubuntu Mono**.

<pre>
{% assign bar = "+------------------------------+------------+" %}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{% for post in site.posts %}
| {{ post.title | printf: "%-28s" }} | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>
