import streamlit as st
import pandas as pd
import time
import json
from db import get_db_connection


def show_admin_dashboard():
    st.markdown("""
        <h1 style='text-align:center; color:#2E86C1;'>🛠️ Admin Control Panel</h1>
        <p style='text-align:center; color:gray;'>Manage users, summaries, paraphrases, and feedback insights.</p>
        <hr style='margin-top: 10px; margin-bottom: 20px;'>
    """, unsafe_allow_html=True)

    # Sidebar Tabs
    tabs = ["Overview", "Summaries", "Paraphrases", "Users", "System Analytics"]
    selected_tab = st.sidebar.radio("🧭 Navigation", tabs)

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # ----------------------------
    # OVERVIEW TAB
    # ----------------------------
    if selected_tab == "Overview":
        st.subheader("📊 Platform Overview")

        try:
            # ---------------------------
            # 🧩 Total Stats Overview
            # ---------------------------
            cursor.execute("SELECT COUNT(*) AS c FROM users")
            total_users = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) AS c FROM summaries")
            total_summaries = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) AS c FROM paraphrases")
            total_paraphrases = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating = 'thumbs_up'")
            total_likes = cursor.fetchone()["c"]

            cursor.execute("SELECT COUNT(*) AS c FROM feedback WHERE rating = 'thumbs_down'")
            total_dislikes = cursor.fetchone()["c"]

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("👥 Users", total_users)
            col2.metric("🧠 Summaries", total_summaries)
            col3.metric("🌀 Paraphrases", total_paraphrases)
            col4.metric("👍 Likes", total_likes)
            col5.metric("👎 Dislikes", total_dislikes)

            # ---------------------------
            # 💬 Feedback Summary Chart
            # ---------------------------
            st.markdown("### 💬 Feedback Summary (Overall Sentiment)")

            cursor.execute("""
                SELECT rating, COUNT(*) AS count
                FROM feedback
                GROUP BY rating;
            """)
            feedback_stats = cursor.fetchall()

            import plotly.express as px

            if feedback_stats:
                df_feedback = pd.DataFrame(feedback_stats)
                df_feedback["rating"] = df_feedback["rating"].replace({
                    "thumbs_up": "👍 Liked",
                    "thumbs_down": "👎 Disliked"
                })

                fig = px.pie(
                    df_feedback,
                    names="rating",
                    values="count",
                    title="Feedback Sentiment Distribution",
                    color="rating",
                    color_discrete_map={"👍 Liked": "green", "👎 Disliked": "red"}
                )
                st.plotly_chart(fig, use_container_width=True)

                # Optional textual insight
                total_feedback = total_likes + total_dislikes
                if total_feedback > 0:
                    like_ratio = (total_likes / total_feedback) * 100
                    st.markdown(f"✅ **{like_ratio:.1f}% of all feedback is positive!**")
                else:
                    st.info("No feedback yet — users haven’t rated any outputs.")
            else:
                st.info("No feedback data available yet.")

        except Exception as e:
            st.error(f"Database error: {e}")

    # ----------------------------
    # SUMMARIES TAB
    # ----------------------------
        # ----------------------------
    # SUMMARIES TAB (Fixed for Unique Keys)
    # ----------------------------
    elif selected_tab == "Summaries":
        st.subheader("🧠 Manage Summaries with Feedback")

        try:
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("🔍 Search by user email or text...")
            with col2:
                feedback_filter = st.selectbox("Filter by feedback", ["All", "Liked", "Disliked", "No Feedback"])

            # include feedback.id for unique keys
            base_query = """
                SELECT s.*, f.id AS feedback_id, f.rating, f.comment
                FROM summaries s
                LEFT JOIN feedback f ON f.output_id = s.id AND f.output_type = 'summary'
            """
            conditions, params = [], []

            if search_query:
                conditions.append("(s.user_email LIKE %s OR s.summary_text LIKE %s OR s.original_text LIKE %s)")
                params.extend([f"%{search_query}%"] * 3)

            if feedback_filter == "Liked":
                conditions.append("f.rating = 'thumbs_up'")
            elif feedback_filter == "Disliked":
                conditions.append("f.rating = 'thumbs_down'")
            elif feedback_filter == "No Feedback":
                conditions.append("f.id IS NULL")

            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
            base_query += " ORDER BY s.created_at DESC LIMIT 100;"

            cursor.execute(base_query, tuple(params))
            data = cursor.fetchall()

            if not data:
                st.info("No summaries found for the selected filter.")
            else:
                for i, row in enumerate(data):
                    # make Streamlit widget keys unique by combining summary_id + feedback_id + index
                    unique_suffix = f"{row['id']}_{row.get('feedback_id', 'none')}_{i}"

                    icon = "👍" if row["rating"] == "thumbs_up" else "👎" if row["rating"] == "thumbs_down" else "⚪"
                    with st.expander(f"{icon} Summary by {row['user_email']} — {row['created_at']}"):
                        st.markdown(f"**🗒️ Original Text:**\n\n{row['original_text']}")

                        edited_summary = st.text_area(
                            "✏️ Edit Summary",
                            row["summary_text"],
                            key=f"sum_edit_{unique_suffix}"
                        )

                        st.markdown(f"**Model Used:** {row['model_used']} | **Length:** {row['summary_length']}")
                        if row.get("rating"):
                            st.markdown(f"**User Feedback:** {'👍 Liked' if row['rating'] == 'thumbs_up' else '👎 Disliked'}")
                        if row.get("comment"):
                            st.info(f"💬 Comment: {row['comment']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Changes", key=f"save_sum_{unique_suffix}"):
                                cursor.execute("UPDATE summaries SET summary_text=%s WHERE id=%s",
                                               (edited_summary, row["id"]))
                                conn.commit()
                                st.success("✅ Summary updated.")
                                time.sleep(0.5)
                                st.rerun()
                        with col2:
                            if st.button("🗑️ Delete Summary", key=f"del_sum_{unique_suffix}"):
                                cursor.execute("DELETE FROM summaries WHERE id=%s", (row["id"],))
                                cursor.execute("DELETE FROM feedback WHERE output_id=%s AND output_type='summary'",
                                               (row["id"],))
                                conn.commit()
                                st.warning("⚠️ Summary and feedback deleted.")
                                time.sleep(0.5)
                                st.rerun()
        except Exception as e:
            st.error(f"Database error: {e}")

    # ----------------------------
    # PARAPHRASES TAB
    # ----------------------------
    elif selected_tab == "Paraphrases":
        st.subheader("🌀 Manage Paraphrases with Feedback")

        try:
            col1, col2 = st.columns([2, 1])
            with col1:
                search_query = st.text_input("🔍 Search by user email or text...")
            with col2:
                feedback_filter = st.selectbox("Filter by feedback", ["All", "Liked", "Disliked", "No Feedback"])

            query = """
                SELECT p.*, f.rating, f.comment
                FROM paraphrases p
                LEFT JOIN feedback f ON f.output_id = p.id AND f.output_type = 'paraphrase'
            """
            conditions, params = [], []

            if search_query:
                conditions.append("(p.user_email LIKE %s OR p.original_text LIKE %s)")
                params.extend([f"%{search_query}%"] * 2)

            if feedback_filter == "Liked":
                conditions.append("f.rating = 'thumbs_up'")
            elif feedback_filter == "Disliked":
                conditions.append("f.rating = 'thumbs_down'")
            elif feedback_filter == "No Feedback":
                conditions.append("f.id IS NULL")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY p.created_at DESC LIMIT 100;"

            cursor.execute(query, tuple(params))
            data = cursor.fetchall()

            if not data:
                st.info("No paraphrases found for selected filters.")
            else:
                for row in data:
                    icon = "👍" if row["rating"] == "thumbs_up" else "👎" if row["rating"] == "thumbs_down" else "⚪"
                    with st.expander(f"{icon} Paraphrase by {row['user_email']} — {row['created_at']}"):
                        st.markdown(f"**🗒️ Original Text:**\n\n{row['original_text']}")
                        try:
                            paraphrases = json.loads(row["paraphrased_options"])
                            selected_idx = st.selectbox("Choose paraphrase:", range(len(paraphrases)),
                                                        format_func=lambda i: paraphrases[i][:80] + "..." if len(paraphrases[i]) > 80 else paraphrases[i],
                                                        key=f"para_sel_{row['id']}")
                            edited_text = st.text_area("✏️ Edit Selected Paraphrase", paraphrases[selected_idx],
                                                       key=f"para_edit_{row['id']}")
                            paraphrases[selected_idx] = edited_text
                        except Exception:
                            paraphrases = []
                            st.error("Invalid paraphrase data format.")

                        if row.get("rating"):
                            st.markdown(f"**User Feedback:** {'👍 Liked' if row['rating'] == 'thumbs_up' else '👎 Disliked'}")
                        if row.get("comment"):
                            st.info(f"💬 Comment: {row['comment']}")

                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("💾 Save Paraphrase", key=f"save_para_{row['id']}"):
                                cursor.execute("UPDATE paraphrases SET paraphrased_options=%s WHERE id=%s",
                                               (json.dumps(paraphrases), row["id"]))
                                conn.commit()
                                st.success("✅ Paraphrase updated.")
                                time.sleep(0.5)
                                st.rerun()
                        with col2:
                            if st.button("🗑️ Delete Paraphrase", key=f"del_para_{row['id']}"):
                                cursor.execute("DELETE FROM paraphrases WHERE id=%s", (row["id"],))
                                cursor.execute("DELETE FROM feedback WHERE output_id=%s AND output_type='paraphrase'", (row["id"],))
                                conn.commit()
                                st.warning("⚠️ Paraphrase and feedback deleted.")
                                time.sleep(0.5)
                                st.rerun()
        except Exception as e:
            st.error(f"Database error: {e}")

    # ----------------------------
    # USER MANAGEMENT TAB (unchanged)
    # ----------------------------
    elif selected_tab == "Users":
        st.subheader("👥 User Management")

        try:
            cursor.execute("SELECT mailid, name, age_group, lang_pre FROM users ORDER BY name ASC;")
            users = cursor.fetchall()
            if not users:
                st.info("No users found.")
            else:
                user_emails = [u["mailid"] for u in users]
                selected_user = st.selectbox("Select a user:", user_emails)

                if selected_user:
                    cursor.execute("SELECT * FROM users WHERE mailid=%s", (selected_user,))
                    user_info = cursor.fetchone()

                    st.markdown(f"### 👤 {user_info['name']}")
                    st.write(f"**Email:** {user_info['mailid']}")
                    st.write(f"**Age Group:** {user_info['age_group'] or 'N/A'}")
                    st.write(f"**Language Preference:** {user_info['lang_pre']}")

                    cursor.execute("SELECT COUNT(*) AS c FROM summaries WHERE user_email=%s", (selected_user,))
                    summary_count = cursor.fetchone()["c"]
                    cursor.execute("SELECT COUNT(*) AS c FROM paraphrases WHERE user_email=%s", (selected_user,))
                    para_count = cursor.fetchone()["c"]

                    st.markdown(f"**🧠 Summaries:** {summary_count} | **🌀 Paraphrases:** {para_count}")

                    st.markdown("---")
                    if st.button("🗑️ Remove User and Data", key=f"del_user_{selected_user}"):
                        cursor.execute("DELETE FROM uploaded_files WHERE user_email=%s", (selected_user,))  # ✅ add this line
                        cursor.execute("DELETE FROM feedback WHERE user_email=%s", (selected_user,))
                        cursor.execute("DELETE FROM summaries WHERE user_email=%s", (selected_user,))
                        cursor.execute("DELETE FROM paraphrases WHERE user_email=%s", (selected_user,))
                        cursor.execute("DELETE FROM users WHERE mailid=%s", (selected_user,))
                        conn.commit()
                        st.warning("⚠️ User and all related data deleted.")
                        time.sleep(0.5)
                        st.rerun()
        except Exception as e:
            st.error(f"Database error: {e}")

    # ----------------------------
    # SYSTEM ANALYTICS TAB
    # ----------------------------
    # ----------------------------
    # SYSTEM ANALYTICS TAB
    # ----------------------------
    elif selected_tab == "System Analytics":
        import plotly.express as px

        st.subheader("📈 System Analytics Dashboard")

        try:
            # Fetch feedback data grouped by date and rating
            cursor.execute("""
                SELECT DATE(created_at) AS date, rating, COUNT(*) AS count
                FROM feedback
                GROUP BY DATE(created_at), rating
                ORDER BY date ASC;
            """)
            fb_trends = cursor.fetchall()

            if fb_trends:
                df_trends = pd.DataFrame(fb_trends)
                fig_trend = px.bar(
                    df_trends,
                    x="date",
                    y="count",
                    color="rating",
                    barmode="group",
                    title="👍 Likes vs 👎 Dislikes Over Time",
                    labels={"date": "Date", "count": "Count", "rating": "Feedback Type"},
                    color_discrete_map={"thumbs_up": "green", "thumbs_down": "red"}
                )
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                st.info("No feedback records yet to display trends.")

            # ----------------------
            # Feedback by Model Used
            # ----------------------
            st.markdown("### 🧠 Feedback by Model Used")
            cursor.execute("""
                SELECT s.model_used AS model, f.rating, COUNT(*) AS count
                FROM feedback f
                JOIN summaries s ON f.output_id = s.id AND f.output_type = 'summary'
                GROUP BY model, rating
                UNION ALL
                SELECT p.model_used AS model, f.rating, COUNT(*) AS count
                FROM feedback f
                JOIN paraphrases p ON f.output_id = p.id AND f.output_type = 'paraphrase'
                GROUP BY model, rating;
            """)
            model_data = cursor.fetchall()

            if model_data:
                df_models = pd.DataFrame(model_data)
                fig_models = px.bar(
                    df_models,
                    x="model",
                    y="count",
                    color="rating",
                    barmode="group",
                    title="🧠 Feedback Distribution by Model",
                    labels={"model": "Model Used", "count": "Feedback Count"},
                    color_discrete_map={"thumbs_up": "green", "thumbs_down": "red"}
                )
                st.plotly_chart(fig_models, use_container_width=True)
            else:
                st.info("No model-based feedback data available yet.")

            # ----------------------
            # Most Active Users (Feedback Count)
            # ----------------------
            st.markdown("### 👥 Most Active Feedback Users")
            cursor.execute("""
                SELECT user_email, COUNT(*) AS total_feedback
                FROM feedback
                GROUP BY user_email
                ORDER BY total_feedback DESC
                LIMIT 10;
            """)
            user_data = cursor.fetchall()

            if user_data:
                df_users = pd.DataFrame(user_data)
                fig_users = px.bar(
                    df_users,
                    x="user_email",
                    y="total_feedback",
                    title="💬 Most Active Feedback Givers",
                    labels={"user_email": "User", "total_feedback": "Feedback Count"},
                    color_discrete_sequence=["#4B9CD3"]
                )
                st.plotly_chart(fig_users, use_container_width=True)
            else:
                st.info("No feedback activity from users yet.")

            st.success("✅ Feedback analytics successfully loaded.")
        except Exception as e:
            st.error(f"Error generating analytics: {e}")


    # ----------------------------
    # LOGOUT BUTTON
    # ----------------------------
    st.sidebar.markdown("---")
    if st.sidebar.button("🔒 Logout as Admin"):
        for key in ["admin_email", "admin_name", "admin_login_mode"]:
            st.session_state.pop(key, None)
        st.session_state.page = "login"
        st.info("Logging out as admin...")
        time.sleep(1)
        st.rerun()

    cursor.close()
    conn.close()
