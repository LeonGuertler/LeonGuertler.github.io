---
layout: page     # use the plain layout; ‘home’ would already list posts below
title:  "👋 Welcome"
---

A minimal AI notebook, rendered entirely in **Ubuntu Mono**.

<!--
ASCII TABLE OF CONTENTS
───────────────────────
This Liquid loop renders at build time, so every new post is listed automatically.
Alignment may be a little ragged for very long titles—that’s life in ASCII 🙂
-->
{% assign bar = "+------------------------------+------------+" %}
{{ bar }}
| Post                         | Date       |
{{ bar }}
{% for post in site.posts %}
| [{{ post.title }}]({{ post.url }}) | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
