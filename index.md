<pre>
+------------------------------------------+------------+
| Post                                     | Date       |
+------------------------------------------+------------+
| <a href="/mindgames-neurips25" style="color:inherit;text-decoration:none;">MindGames (NeurIPS25 competition)</a> | 2025-07-14 |
| <a href="/how-we-built-textarena" style="color:inherit;text-decoration:none;">How we built TextArena</a>            | 2025-07-12 |
| <a href="/hello-world" style="color:inherit;text-decoration:none;">Hello, world</a>                          | 2025-07-11 |
+------------------------------------------+------------+
</pre>













<pre>
{% assign bar = "+------------------------------------------+------------+" %}
{{ bar }}
| Post                                     | Date       |
{{ bar }}
{% for post in site.posts %}
{%  comment %} pad each title to 28 chars {% endcomment %}
{% assign padded = post.title | append: "                            " | slice: 0, 40 %}
| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ padded }}</a> | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>




<pre>
{%- assign bar = "+------------------------------------------+------------+" -%}
{{ bar }}
| Post                                      | Date       |
{{ bar }}
{%- assign pad = "                                          " -%}  {# 42 spaces #}
{%- for post in site.posts -%}
| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ post.title | append: pad | slice: 0, 42 }}</a> | {{ post.date | date: "%Y-%m-%d" }} |

{%- endfor -%}
{{ bar }}
</pre>


<pre>
{%- assign bar = "+------------------------------------------+------------+" -%}
{{ bar }}
| Post                                      | Date       |
{{ bar }}
{% assign pad = "                                          " %}   <!-- 42 spaces -->
{% for post in site.posts %}
| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ post.title | append: pad | slice: 0, 42 }}</a> | {{ post.date | date: "%Y-%m-%d" }} |
{% endfor %}
{{ bar }}
</pre>



<pre>
{%- assign bar = "+------------------------------------------+------------+" -%}
{{ bar }}
| Post                                      | Date       |
{{ bar }}
{%- assign pad = "                                          " -%}   <!-- 42 spaces -->
{%- for post in site.posts -%}
| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ post.title | append: pad | slice: 0, 42 }}</a> | {{ post.date | date: "%Y-%m-%d" }} |
{%- endfor -%}
{{ bar }}
</pre>


<pre>
{%- assign bar = "+------------------------------------------+------------+" -%}
{{ bar }}
| Post                                      | Date       |
{{ bar }}
{%- assign pad = "                                          " -%}   {# 42 spaces #}
{%- for post in site.posts -%}
| <a href="{{ post.url | relative_url }}" style="color:inherit;text-decoration:none;">{{ post.title | append: pad | slice: 0, 42 }}</a> | {{ post.date | date: "%Y-%m-%d" }} |
{%- endfor -%}
{{ bar }}
</pre>
