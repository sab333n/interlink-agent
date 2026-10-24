import asyncio
import socket
import unittest
from unittest import mock

import agent
import queue_runner
from routes import inline_apply
from services import answers


def field(fid, label, kind="input", required=True, options=None, value=""):
    return {"id": fid, "label": label, "kind": kind, "required": required, "options": options or [], "value": value}


class SandboxTest(unittest.TestCase):
    def test_network_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
            socket.create_connection(("example.com", 80))

    def test_no_llm_or_supabase_configured(self):
        self.assertFalse(agent.llm_configured())
        self.assertIsNone(answers.chat.llm_config())
        self.assertIsNone(answers.admin_config())


class PersonalSiteTest(unittest.TestCase):
    def test_keeps_real_sites_and_profile_links(self):
        for url in ("https://sabin.dev", "x.com/sabin", "github.com/sab333n", "linkedin.com/in/someone"):
            self.assertEqual(agent.personal_site(url), url)

    def test_blanks_social_homepages_and_junk(self):
        for url in ("x.com", "https://x.com/", "www.x.com", "github.com", "not a url", "localhost", "", None):
            self.assertEqual(agent.personal_site(url), "", url)


class AskUserTest(unittest.TestCase):
    def test_matches_questions_only_the_applicant_can_answer(self):
        for q in (
            "How do you pronounce your name? (e.g., 'An-na' or 'Ah-na')",
            "Undergraduate GPA",
            "Do you have a Duolingo account? If yes, what is your username?",
            "Have you ever been convicted of a felony?",
            "What is your GitHub handle?",
        ):
            self.assertTrue(agent.ASK_USER.search(q), q)

    def test_leaves_ordinary_questions_alone(self):
        for q in ("Are you legally authorized to work in the United States?", "Why do you want to work here?",
                  "Email", "How did you hear about this job?"):
            self.assertFalse(agent.ASK_USER.search(q), q)


class DecideYesNoTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(agent.PROFILE, {"work_auth": "Yes", "sponsorship": "No", "relocation": "Yes"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_answers_from_profile(self):
        self.assertEqual(agent.decide_yes_no("Are you authorized to work in the US?"), "Yes")
        self.assertEqual(agent.decide_yes_no("Will you now or in the future require sponsorship?"), "No")
        self.assertEqual(agent.decide_yes_no("Have you previously worked at Acme?"), "No")

    def test_never_guesses_personal_history(self):
        self.assertIsNone(agent.decide_yes_no("Have you ever been convicted of a felony?"))

    def test_blank_profile_fields_leave_the_question_unanswered(self):
        with mock.patch.dict(agent.PROFILE, {"work_auth": "", "sponsorship": "", "relocation": ""}):
            self.assertIsNone(agent.decide_yes_no("Are you authorized to work in the US?"))
            self.assertIsNone(agent.decide_yes_no("Will you now or in the future require sponsorship?"))
            self.assertIsNone(agent.decide_yes_no("Are you willing to relocate?"))

    def test_no_values_still_answer_no(self):
        with mock.patch.dict(agent.PROFILE, {"work_auth": "No", "sponsorship": "Yes", "relocation": "no"}):
            self.assertEqual(agent.decide_yes_no("Are you authorized to work in the US?"), "No")
            self.assertEqual(agent.decide_yes_no("Do you require visa sponsorship?"), "Yes")
            self.assertEqual(agent.decide_yes_no("Are you willing to relocate?"), "No")

    def test_ask_grok_skips_the_model_for_ask_user_questions(self):
        with mock.patch.object(agent, "grok_complete", side_effect=AssertionError("model called")):
            self.assertEqual(agent.ask_grok("Undergraduate GPA"), "")


class QueueRunnerTest(unittest.TestCase):
    def test_graduation_question_reads_profile(self):
        # Regression: `grad` was used before assignment and raised NameError.
        with mock.patch.dict(queue_runner.PROFILE, {"graduation_year": "2028"}):
            self.assertEqual(queue_runner.question_answer("What is your expected graduation timeline?"), "2028")

    def test_blank_profile_fields_leave_sentence_questions_unanswered(self):
        blank = {"work_auth": "", "sponsorship": "", "relocation": "", "visa_status": ""}
        with mock.patch.dict(queue_runner.PROFILE, blank), mock.patch.dict(agent.PROFILE, blank):
            for q in ("Are you legally authorized to work in the United States?",
                      "Will you require H-1B sponsorship now or in the future?",
                      "Are you willing to work from the office 3 days a week?"):
                self.assertIsNone(queue_runner.question_answer(q), q)

    def test_ai_answer_never_asks_the_model_about_ask_user_fields(self):
        with mock.patch.object(queue_runner.agent, "grok_complete", side_effect=AssertionError("model called")):
            self.assertEqual(queue_runner.ai_answer("Do you have a Duolingo account?", {"company": "Duolingo"}), "")


class InlineProfileTest(unittest.TestCase):
    def test_use_profile_drops_junk_website(self):
        inline_apply._use_profile({"first_name": "Sabin", "portfolio": "x.com"}, None)
        self.assertEqual(queue_runner.PROFILE["portfolio"], "")
        self.assertEqual(queue_runner.PROFILE["first_name"], "Sabin")


class AnswerHelpersTest(unittest.TestCase):
    def test_placeholder_values(self):
        for v in ("N/A", "n/a", "None", "Not Specified", "-", "TBD"):
            self.assertTrue(answers.PLACEHOLDER.match(v), v)
        self.assertFalse(answers.PLACEHOLDER.match("3.7"))

    def test_is_essay(self):
        self.assertTrue(answers._is_essay(field("a", "Why are you interested in Duolingo?", "textarea")))
        self.assertTrue(answers._is_essay(field("b", "What excites you about this role?")))
        self.assertFalse(answers._is_essay(field("c", "Undergraduate GPA", "textarea")))
        self.assertFalse(answers._is_essay(field("d", "Preferred name, if different?")))
        self.assertFalse(answers._is_essay(field("e", "Why us?", "select", options=["A", "B"])))

    def test_clean_maps_answers_onto_options(self):
        self.assertEqual(answers._clean(field("a", "Q", "select", options=["Yes", "No"]), "yes"), "Yes")
        self.assertIsNone(answers._clean(field("a", "Q", "select", options=["Yes", "No"]), "Maybe"))
        self.assertEqual(answers._clean(field("b", "Agree", "checkbox"), "Yes"), "true")
        self.assertEqual(
            answers._clean(field("c", "Terms", "checkboxes", options=["Summer 2027", "Fall 2027", "Spring 2028"]),
                           "summer 2027, Spring 2028"),
            "Summer 2027, Spring 2028",
        )
        self.assertIsNone(answers._clean(field("d", "Q"), "  "))


class FallbackTest(unittest.TestCase):
    profile = {**agent._PROFILE_DEFAULTS, "first_name": "Sabin", "github": "https://github.com/sab333n",
               "work_auth": "Yes", "gpa": "Not Specified"}
    job = {"company": "Duolingo", "role": "Data Scientist, Intern", "location": "Pittsburgh, PA"}

    def hint(self, f, profile=None):
        return answers._fallback(f, self.job, profile or self.profile)

    def test_fills_facts_from_profile(self):
        self.assertEqual(self.hint(field("a", "Preferred First Name"))["answer"], "Sabin")
        self.assertEqual(self.hint(field("b", "Website"))["answer"], "https://github.com/sab333n")
        auth = self.hint(field("c", "Are you legally authorized to work in the United States?", "select",
                               options=["Yes", "No"]))
        self.assertEqual(auth["answer"], "Yes")

    def test_blank_work_auth_sponsorship_relocation_stay_blank(self):
        blank = {**self.profile, "work_auth": "", "sponsorship": "", "relocation": ""}
        for label in ("Are you legally authorized to work in the United States?",
                      "Will you require sponsorship?", "Are you willing to relocate?"):
            self.assertIsNone(self.hint(field("q", label, "select", options=["Yes", "No"]), blank), label)

    def test_not_authorized_text_answers_no(self):
        hint = self.hint(field("q", "Are you legally authorized to work in the US?", "select", options=["Yes", "No"]),
                         {**self.profile, "work_auth": "Not authorized"})
        self.assertEqual(hint["answer"], "No")

    def test_gpa_only_from_a_real_profile_value(self):
        self.assertIsNone(self.hint(field("a", "Undergraduate GPA")))
        self.assertEqual(self.hint(field("a", "Undergraduate GPA"), {**self.profile, "gpa": "3.7"})["answer"], "3.7")

    def test_never_guesses(self):
        for f in (
            field("a", "How do you pronounce your name?"),
            field("b", "Do you have a Duolingo account? If yes, what is your username?"),
            field("c", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
            field("d", "Do you have a security clearance?", "select", options=["Yes", "No"]),
            field("e", "Why are you interested in Duolingo?", "textarea"),
        ):
            self.assertIsNone(self.hint(f), f["label"])


class SuggestWithoutModelTest(unittest.TestCase):
    def test_fills_facts_and_leaves_the_rest_blank(self):
        profile = {**FallbackTest.profile}
        fields = [
            field("name", "Preferred First Name"),
            field("gpa", "Undergraduate GPA"),
            field("pron", "How do you pronounce your name?"),
            field("why", "Why are you interested in Duolingo?", "textarea"),
            field("impact", "How will your personal experiences make an impact at Duolingo?", "textarea"),
            field("felony", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
            field("done", "Email", value="already@filled.com"),
        ]
        out = asyncio.run(answers.suggest(fields, FallbackTest.job, profile, "local"))
        self.assertEqual(out, {"name": {"answer": "Sabin", "source": "fact"}})


USER = "0b9f3c2e-1a2b-4c5d-8e9f-001122334455"
DUOLINGO = {"company": "Duolingo", "role": "Data Scientist, Intern"}
REVIEW_FIELDS = [
    field("r", "Resume", "file"),
    field("pron", "How do you pronounce your name? (e.g., 'An-na' or 'Ah-na')"),
    field("gpa", "Undergraduate GPA"),
    field("acct", "Do you have a Duolingo account? If yes, what is your username?"),
    field("why", "Why are you interested in Duolingo?", "textarea"),
    field("fel", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
]


class ReviewedPairsTest(unittest.TestCase):
    def test_keeps_real_answers_by_label(self):
        pairs = inline_apply.reviewed_pairs(REVIEW_FIELDS, {
            "r": "resume.pdf", "pron": "SAH-bin", "gpa": "3.7", "acct": " ", "why": "Select…",
            "fel": "no", "unknown": "x",
        })
        self.assertEqual(pairs, [
            ("How do you pronounce your name? (e.g., 'An-na' or 'Ah-na')", "SAH-bin"),
            ("Undergraduate GPA", "3.7"),
            ("Have you ever been convicted of a felony?", "No"),
        ])


class HandOffTest(unittest.TestCase):
    """Answers reviewed in InterLink come back verbatim when the extension fills the employer's form."""

    def remember(self, pairs, company="Duolingo"):
        saved = []

        async def fake_db(method, path, body=None, prefer=""):
            saved.extend(body)

        with mock.patch.object(answers, "admin_config", return_value=("https://db", "key")), \
                mock.patch.object(answers, "db", side_effect=fake_db):
            asyncio.run(answers.remember(USER, company, pairs))
        return saved

    def test_remember_stores_exact_and_company_neutral_keys(self):
        rows = self.remember([("Why are you interested in Duolingo?", "Because...")])
        self.assertEqual({r["question"] for r in rows},
                         {"why are you interested in {company}?", "why are you interested in duolingo?"})
        self.assertEqual(self.remember([("  ", "x"), ("GPA", " ")]), [])

    def test_exact_answers_are_reused_without_a_model(self):
        rows = self.remember(inline_apply.reviewed_pairs(REVIEW_FIELDS, {
            "pron": "SAH-bin", "gpa": "3.7", "acct": "No", "why": "I build data tools for learners.", "fel": "No",
        }))

        async def past(_user):
            return [{"question": r["question"], "answer": r["answer"]} for r in rows]

        employer_form = [
            field("e1", "How do you pronounce your name? (e.g., 'An-na' or 'Ah-na') *"),
            field("e2", "Undergraduate GPA"),
            field("e3", "Do you have a Duolingo account? If yes, what is your username?"),
            field("e4", "Why are you interested in Duolingo?", "textarea"),
            field("e5", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
        ]
        with mock.patch.object(answers, "load_past", side_effect=past):
            out = asyncio.run(answers.suggest(employer_form, DUOLINGO, {**agent._PROFILE_DEFAULTS}, USER))
        self.assertEqual({k: v["answer"] for k, v in out.items()}, {
            "e1": "SAH-bin", "e2": "3.7", "e3": "No", "e4": "I build data tools for learners.", "e5": "No",
        })
        self.assertTrue(all(v["source"] == "past" for v in out.values()))

    def test_company_specific_answers_are_not_copied_to_other_employers(self):
        rows = self.remember([("Why are you interested in Duolingo?", "Duolingo's mission...")])

        async def past(_user):
            return rows

        with mock.patch.object(answers, "load_past", side_effect=past):
            out = asyncio.run(answers.suggest([field("w", "Why are you interested in Stripe?", "textarea")],
                                              {"company": "Stripe"}, {**agent._PROFILE_DEFAULTS}, USER))
        self.assertEqual(out, {})

    def test_extension_fill_returns_the_saved_answers(self):
        from routes import extension

        rows = self.remember([("Undergraduate GPA", "3.7"), ("Do you have a Duolingo account? If yes, what is your username?", "No")])

        async def past(_user):
            return rows

        async def info(_user):
            return {"profile": {"first_name": "Sabin"}}

        req = extension.FillRequest(fields=[
            extension.Field(id="a", kind="input", label="First Name", required=True),
            extension.Field(id="b", kind="input", label="Undergraduate GPA *", required=True),
            extension.Field(id="c", kind="input", label="Do you have a Duolingo account? If yes, what is your username?", required=True),
        ], job=DUOLINGO)
        with mock.patch.object(answers, "load_past", side_effect=past), \
                mock.patch.object(extension, "_apply_info", side_effect=info):
            out = asyncio.run(extension.fill(req, user=USER))["answers"]
        self.assertEqual({k: v["answer"] for k, v in out.items()}, {"a": "Sabin", "b": "3.7", "c": "No"})


if __name__ == "__main__":
    unittest.main()
